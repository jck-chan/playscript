import re
import time
import threading
from dataclasses import dataclass
from typing import Literal
import keyboard
from pydantic import BaseModel


class Note(BaseModel):
    name: str
    key: str | None
    remarks: str = ""


class KeyMapConfig(BaseModel):
    """Ordered physical key slots for the sheet (validated as a unit)."""

    type: Literal["key_map"] = "key_map"
    notes: list[Note]

# One character: a run of only this in the sheet ties / extends the previous held key
# (e.g. "-" -> one step, "---" -> three steps). Must not match any ``KeyInfo.name``.
TIE_CHAR = "-"

# Keys you want to cycle through (in order)
KEYS = KeyMapConfig(
    notes=[
        Note(name=";", key=None),
        Note(name="1.", key="y"),
        Note(name="2.", key="u"),
        Note(name="3.", key="i"),
        Note(name="4.", key="o"),
        Note(name="5.", key="p"),
        Note(name="6.", key="h"),
        Note(name="7.", key="j"),
        Note(name="1", key="k"),
        Note(name="2", key="l"),
        Note(name="3", key=";"),
        Note(name="4", key="n"),
        Note(name="5", key="m"),
        Note(name="6", key=","),
        Note(name="7", key="."),
        Note(name="1'", key="/"),
    ]
).notes

# str of key name separated by spaces or line breaks.
# Speed regions: ``<x2> ... </x>`` — only wrapped notes use that factor.
# Parallel voices in ``<p>``: separate with ``|`` (outside ``<x>…</x>``). Playback uses a merged time schedule:
# each voice is simulated like serial (own ``speed_mul`` per step), events get absolute times, then sorted and played.
with open("./sheets/canon.txt", "rt") as f:
    text_sheet: str = f.read()

# Total length of one beat (seconds), before speed scaling.
single_beat_seconds = 0.35

# Release the physical key this many seconds *before* the beat ends (staccato gap).
# Scaled by speed like the beat (shorter beats → proportionally shorter early release).
# If the next step is a tie, the key stays down for the full beat (no early release).
release_before_beat_end = 0.03

# Hotkey to start the script (press once)
START_HOTKEY = "alt+p"


def _is_tie_run(token: str, tie_char: str) -> bool:
    """True if ``token`` is one or more ``tie_char`` only (e.g. ``---`` when tie is ``-``)."""
    return bool(token) and all(c == tie_char for c in token)


@dataclass(frozen=True)
class SheetKeyStep:
    key_index: int


@dataclass(frozen=True)
class SheetTieStep:
    """Extend the previous physical key for one hold interval (not a ``KeyInfo``)."""


@dataclass(frozen=True)
class SheetTimedStep:
    """One key or tie beat with its own speed (from ``<xN>…</x>`` scope)."""

    speed_mul: float
    step: SheetKeyStep | SheetTieStep


@dataclass(frozen=True)
class SheetParallel:
    """Several melodies in ``<p>``; each voice has its own timeline, merged by wall-clock."""

    voices: tuple[tuple[SheetTimedStep, ...], ...]


SheetItem = SheetTimedStep | SheetParallel

# Refcount so parallel voices can share a physical key without one releasing the other's hold.
_key_press_refcount: dict[str, int] = {}
_keyboard_lock = threading.Lock()


def _keyboard_press_key(key: str) -> None:
    with _keyboard_lock:
        _key_press_refcount[key] = _key_press_refcount.get(key, 0) + 1
        if _key_press_refcount[key] == 1:
            keyboard.press(key)


def _keyboard_release_key(key: str) -> None:
    with _keyboard_lock:
        n = _key_press_refcount.get(key, 0)
        if n <= 0:
            return
        _key_press_refcount[key] = n - 1
        if _key_press_refcount[key] == 0:
            del _key_press_refcount[key]
            keyboard.release(key)


def release_all_keys() -> None:
    with _keyboard_lock:
        for k in list(_key_press_refcount.keys()):
            keyboard.release(k)
        _key_press_refcount.clear()


_X_OPEN_RE = re.compile(
    r"<x\s*([0-9]+\.?[0-9]*|[0-9]*\.[0-9]+)\s*>",
    re.IGNORECASE,
)
_X_CLOSE_RE = re.compile(r"</x\s*>", re.IGNORECASE)
_P_BLOCK_RE = re.compile(r"<p\s*>(.*?)</p\s*>", re.IGNORECASE | re.DOTALL)


def _split_parallel_voice_segments(s: str) -> list[str]:
    """Split on ``|`` only outside ``<xN>…</x>``."""
    parts: list[str] = []
    cur: list[str] = []
    i = 0
    n = len(s)
    depth_x = 0
    while i < n:
        if depth_x == 0 and s[i] == "|":
            seg = "".join(cur).strip()
            if seg:
                parts.append(seg)
            cur = []
            i += 1
            while i < n and s[i] == " ":
                i += 1
            continue
        mo = _X_OPEN_RE.match(s, i)
        if mo is not None:
            cur.append(s[i : mo.end()])
            depth_x += 1
            i = mo.end()
            continue
        mc = _X_CLOSE_RE.match(s, i)
        if mc is not None:
            cur.append(s[i : mc.end()])
            depth_x -= 1
            i = mc.end()
            continue
        cur.append(s[i])
        i += 1
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def _find_matching_x_close(s: str, inner_start: int) -> tuple[int, int]:
    """Return ``(index_of_close_tag, index_after_close)`` for the first ``</x>`` matching opens."""
    depth = 1
    i = inner_start
    while depth > 0:
        mo = _X_OPEN_RE.search(s, i)
        mc = _X_CLOSE_RE.search(s, i)
        if mc is None:
            raise ValueError("Unclosed <xN>… region (missing </x>)")
        if mo is not None and mo.start() < mc.start():
            depth += 1
            i = mo.end()
        else:
            depth -= 1
            if depth == 0:
                return (mc.start(), mc.end())
            i = mc.end()
    raise RuntimeError("unreachable")


def _tokens_to_key_tie_steps(
    tokens: list[str],
    name_to_index: dict[str, int],
    tie_char: str,
) -> list[SheetKeyStep | SheetTieStep]:
    steps: list[SheetKeyStep | SheetTieStep] = []
    for token in tokens:
        if token == "|":
            continue
        if _is_tie_run(token, tie_char):
            steps.extend(SheetTieStep() for _ in range(len(token)))
            continue
        if token not in name_to_index:
            known = ", ".join(sorted(name_to_index, key=lambda s: (len(s), s)))
            raise ValueError(f"Unknown sheet token {token!r}. Known names: {known}")
        steps.append(SheetKeyStep(name_to_index[token]))
    return steps


def _plain_text_to_timed_steps(
    text: str,
    speed_mul: float,
    name_to_index: dict[str, int],
    tie_char: str,
) -> list[SheetTimedStep]:
    out: list[SheetTimedStep] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for st in _tokens_to_key_tie_steps(line.split(), name_to_index, tie_char):
            out.append(SheetTimedStep(speed_mul=speed_mul, step=st))
    return out


def _parse_timed_string(
    s: str,
    speed_mul: float,
    name_to_index: dict[str, int],
    tie_char: str,
) -> list[SheetTimedStep]:
    """Parse notes/ties and nested ``<xN>inner</x>``; ``speed_mul`` applies outside inner regions."""
    result: list[SheetTimedStep] = []
    pos = 0
    n = len(s)
    while pos < n:
        m = _X_OPEN_RE.search(s, pos)
        if m is None:
            result.extend(_plain_text_to_timed_steps(s[pos:], speed_mul, name_to_index, tie_char))
            break
        if m.start() > pos:
            result.extend(
                _plain_text_to_timed_steps(
                    s[pos : m.start()], speed_mul, name_to_index, tie_char
                )
            )
        factor = float(m.group(1))
        if factor <= 0:
            raise ValueError(f"Speed factor in {m.group(0)!r} must be positive")
        inner_start = m.end()
        close_start, after_close = _find_matching_x_close(s, inner_start)
        inner = s[inner_start:close_start]
        result.extend(
            _parse_timed_string(inner, factor, name_to_index, tie_char)
        )
        pos = after_close
    return result


def _parse_plain_sheet_chunk(
    chunk: str,
    name_to_index: dict[str, int],
    tie_char: str,
) -> list[SheetTimedStep]:
    return _parse_timed_string(chunk, 1.0, name_to_index, tie_char)


def _parse_parallel_inner(
    inner: str,
    name_to_index: dict[str, int],
    tie_char: str,
) -> SheetParallel:
    normalized = " ".join(inner.split())
    parts = [p.strip() for p in _split_parallel_voice_segments(normalized) if p.strip()]
    if not parts:
        raise ValueError(
            "<p> block must contain at least one voice (non-empty segment)."
        )
    voices: list[tuple[SheetTimedStep, ...]] = []
    for p in parts:
        voices.append(tuple(_parse_timed_string(p, 1.0, name_to_index, tie_char)))
    return SheetParallel(voices=tuple(voices))


def _parse_sheet(text: str, keys: list[Note], tie_char: str) -> list[SheetItem]:
    """Parse sheet: ``<p>…</p>`` parallel blocks, ``<xN>…</x>`` speed regions, notes."""
    if len(tie_char) != 1:
        raise ValueError("tie_char must be exactly one character (set TIE_CHAR).")
    name_to_index = {k.name: i for i, k in enumerate(keys)}
    for k in keys:
        if k.name == tie_char:
            raise ValueError(
                f"KeyInfo name {tie_char!r} collides with TIE_CHAR; rename the key or change TIE_CHAR."
            )
    steps: list[SheetItem] = []
    pos = 0
    for m in _P_BLOCK_RE.finditer(text):
        if m.start() > pos:
            steps.extend(
                _parse_plain_sheet_chunk(text[pos : m.start()], name_to_index, tie_char)
            )
        steps.append(_parse_parallel_inner(m.group(1), name_to_index, tie_char))
        pos = m.end()
    if pos < len(text):
        steps.extend(_parse_plain_sheet_chunk(text[pos:], name_to_index, tie_char))
    return steps


sheet_steps: list[SheetItem] = _parse_sheet(text_sheet, KEYS, TIE_CHAR)

# pause_event: set means paused; cleared means running
pause_event = threading.Event()
current_key = None
shutdown_event = threading.Event()


def release_current_if_any() -> None:
    """Release the single key used by the serial melody (not other parallel holds)."""
    global current_key
    if current_key is None:
        return
    _keyboard_release_key(current_key)
    current_key = None


# Below this, use a short busy-wait so Windows' coarse ``sleep()`` (~10–16 ms typical)
# does not swallow tiny gaps (e.g. ``release_before_beat_end``).
_WAIT_SPIN_SEC = 0.001


def wait_or_pause(seconds: float) -> bool:
    """Wait up to ``seconds``, but return early if paused/shutting down."""
    if seconds <= 0:
        return True
    deadline = time.perf_counter() + seconds
    while True:
        if shutdown_event.is_set():
            return False
        if pause_event.is_set():
            return False
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return True
        if remaining > _WAIT_SPIN_SEC:
            # Sleep almost all of it; leave a small tail for accurate wake-up.
            time.sleep(remaining - _WAIT_SPIN_SEC)
        # Spin only the last ~2 ms (negligible CPU; much better than 15 ms sleep grain).


def toggle_pause() -> None:
    """Toggle pause/resume when the start hotkey is pressed."""
    global current_key
    if pause_event.is_set():
        pause_event.clear()
        print("\nResumed.")
    else:
        pause_event.set()
        release_current_if_any()
        release_all_keys()
        current_key = None
        print("\nPaused.")


def press_and_hold(key: str) -> None:
    """Press (and keep held) a key until we release it on the next switch."""
    global current_key
    if current_key is not None and current_key != key:
        _keyboard_release_key(current_key)
    current_key = key
    _keyboard_press_key(key)


def release_current() -> None:
    """Backward-compatible wrapper."""
    release_current_if_any()


def _next_item_after(step: int) -> SheetItem | None:
    j = step + 1
    if j < 0 or j >= len(sheet_steps):
        return None
    return sheet_steps[j]


def _effective_hold_seconds(speed_mul: float) -> float:
    return single_beat_seconds / speed_mul


def _effective_release_before_beat_end(speed_mul: float) -> float:
    return max(0.0, release_before_beat_end) / speed_mul


def _beat_press_and_tail(hold_sec: float, speed_mul: float) -> tuple[float, float]:
    """Key-down part and key-up tail of a beat; tail = early release / silence before next beat."""
    early = min(_effective_release_before_beat_end(speed_mul), hold_sec)
    press_part = hold_sec - early
    tail = early
    return press_part, tail


# (absolute_time_s, sort_prio, voice_idx, physical_key, "press"|"release")
# sort_prio: release before press at same timestamp; then voice_idx for stability.
_SchedEvent = tuple[float, int, int, str, Literal["press", "release"]]


def _schedule_voice_events(
    steps: tuple[SheetTimedStep, ...],
    voice_idx: int,
) -> list[_SchedEvent]:
    """Build press/release events with wall times (same rules as the serial player for one line)."""
    out: list[_SchedEvent] = []
    t = 0.0
    vc_held: str | None = None

    def emit_release(at: float, key: str) -> None:
        out.append((at, 0, voice_idx, key, "release"))

    def emit_press(at: float, key: str) -> None:
        out.append((at, 1, voice_idx, key, "press"))

    for i, item in enumerate(steps):
        speed_mul = item.speed_mul
        hold_sec = _effective_hold_seconds(speed_mul)
        press_part, tail = _beat_press_and_tail(hold_sec, speed_mul)
        nxt = steps[i + 1] if i + 1 < len(steps) else None
        nxt_tie = nxt is not None and isinstance(nxt.step, SheetTieStep)

        if isinstance(item.step, SheetTieStep):
            if nxt_tie:
                t += hold_sec
            else:
                if vc_held is not None:
                    emit_release(t + press_part, vc_held)
                    vc_held = None
                t += hold_sec
            continue

        assert isinstance(item.step, SheetKeyStep)
        ki = KEYS[item.step.key_index]
        if ki.key is None:
            if vc_held is not None:
                emit_release(t, vc_held)
                vc_held = None
            t += hold_sec
            continue

        if vc_held is not None and vc_held != ki.key:
            emit_release(t, vc_held)
            vc_held = None
        vc_held = ki.key
        emit_press(t, ki.key)

        continues_into_tie = nxt_tie
        if continues_into_tie:
            t += hold_sec
        else:
            emit_release(t + press_part, ki.key)
            vc_held = None
            t += hold_sec

    if vc_held is not None:
        emit_release(t, vc_held)

    return out


def _play_parallel(
    voices: tuple[tuple[SheetTimedStep, ...], ...],
) -> None:
    """Merge precomputed per-voice schedules and play by absolute time (not column-locked)."""
    merged: list[_SchedEvent] = []
    for v, steps in enumerate(voices):
        merged.extend(_schedule_voice_events(steps, v))
    merged.sort(key=lambda e: (e[0], e[1], e[2]))

    t_play = 0.0
    for t_abs, _prio, _vidx, key, kind in merged:
        if pause_event.is_set() or shutdown_event.is_set():
            return
        dt = t_abs - t_play
        if dt > 0 and not wait_or_pause(dt):
            return
        t_play = t_abs
        if kind == "press":
            _keyboard_press_key(key)
        else:
            _keyboard_release_key(key)


def main() -> None:
    n_keys = sum(
        1
        for x in sheet_steps
        if isinstance(x, SheetTimedStep) and isinstance(x.step, SheetKeyStep)
    )
    n_ties = sum(
        1
        for x in sheet_steps
        if isinstance(x, SheetTimedStep) and isinstance(x.step, SheetTieStep)
    )
    n_par = sum(1 for x in sheet_steps if isinstance(x, SheetParallel))
    print("Auto key hold tapper ready (start hotkey toggles pause/resume).")
    print(
        f"Sheet: {len(sheet_steps)} top-level items ({n_keys} notes, {n_ties} ties, "
        f"{n_par} <p> blocks)"
    )
    print(
        f"Tie character in sheet: {TIE_CHAR!r} (runs like {TIE_CHAR * 3} = 3 tie steps)"
    )
    print(f"Key definitions: {KEYS}")
    print(
        f"Beat: {single_beat_seconds}s (÷N inside <xN>…</x>); "
        f"release {release_before_beat_end}s before beat end (scaled)"
    )
    print(f"Start hotkey: {START_HOTKEY}")
    print("Exit: Ctrl+C\n")

    # Note: On Windows, this may require running the script as Administrator.
    keyboard.add_hotkey(START_HOTKEY, toggle_pause)

    # Start paused until the first Alt+S press.
    pause_event.set()

    try:
        while not shutdown_event.is_set():
            # If paused, make sure nothing is held.
            if pause_event.is_set():
                global current_key
                release_current_if_any()
                release_all_keys()
                current_key = None
                time.sleep(0.05)
                continue

            for i, item in enumerate(sheet_steps):
                step = i + 1
                if pause_event.is_set() or shutdown_event.is_set():
                    break

                if isinstance(item, SheetParallel):
                    lens = [len(v) for v in item.voices]
                    print(
                        f"[{step}] <p> parallel: {len(item.voices)} voices, steps {lens} (time-merged)"
                    )
                    _play_parallel(item.voices)
                    continue

                assert isinstance(item, SheetTimedStep)
                speed_mul = item.speed_mul
                hold_sec = _effective_hold_seconds(speed_mul)
                press_part, tail = _beat_press_and_tail(hold_sec, speed_mul)
                nxt = _next_item_after(i)
                nxt_tie = (
                    isinstance(nxt, SheetTimedStep) and isinstance(nxt.step, SheetTieStep)
                )

                if isinstance(item.step, SheetTieStep):
                    # Tie: extend previous key for one beat; early release only when tie ends.
                    print(f"[{step}] -")
                    tie_continues = nxt_tie
                    if tie_continues:
                        if not wait_or_pause(hold_sec):
                            break
                    else:
                        if not wait_or_pause(press_part):
                            break
                        release_current_if_any()
                        if tail > 0 and not wait_or_pause(tail):
                            break
                    continue

                idx = item.step.key_index
                ki = KEYS[idx]
                print(
                    f"[{step}] {ki.name}  ({ki.key})"
                    f"{" # " + ki.remarks if ki.remarks else ""}"
                )

                if ki.key is not None:
                    press_and_hold(ki.key)
                else:
                    # Rest (e.g. name ";") — nothing held for this step.
                    release_current_if_any()

                continues_into_tie = ki.key is not None and nxt_tie
                if continues_into_tie:
                    if not wait_or_pause(hold_sec):
                        break
                else:
                    if not wait_or_pause(press_part):
                        break
                    release_current_if_any()
                    if tail > 0 and not wait_or_pause(tail):
                        break
            else:
                # Entire sheet played without ``break`` (no pause/shutdown mid-item).
                print(
                    "\nEnd of sheet. Paused — press start hotkey to play again."
                )
                pause_event.set()

            if shutdown_event.is_set():
                break
    except KeyboardInterrupt:
        shutdown_event.set()
    finally:
        release_current_if_any()
        release_all_keys()
        keyboard.unhook_all_hotkeys()
        print("Stopped.")


if __name__ == "__main__":
    main()
