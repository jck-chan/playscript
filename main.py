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
# Inline speed: ``<x2>`` ``<x0.5>`` — effective beat = single_beat_seconds / factor.
# Parallel voices: ``<p> voice0 / voice1 </p>`` — one column per beat, ``/`` separates parts.
# Lines starting with ``/`` are ignored (visual separators only).
with open("./sheets/canon.txt", "rt") as f:
    text_sheet: str = f.read()

# Total length of one beat (seconds), before speed scaling.
single_beat_seconds = 0.5

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
class SheetSpeedMul:
    """Playback speed multiplier: effective beat = single_beat_seconds / factor."""

    factor: float


@dataclass(frozen=True)
class SheetParallel:
    """Several melodies played together; ``voices`` share each hold column (beat)."""

    voices: tuple[tuple["SheetItem", ...], ...]


SheetItem = SheetKeyStep | SheetTieStep | SheetSpeedMul | SheetParallel

# Keys currently sent as pressed to the OS (serial + parallel).
_physically_down: set[str] = set()


def _keyboard_press_key(key: str) -> None:
    if key not in _physically_down:
        keyboard.press(key)
        _physically_down.add(key)


def _keyboard_release_key(key: str) -> None:
    if key in _physically_down:
        keyboard.release(key)
        _physically_down.discard(key)


def release_all_keys() -> None:
    for k in list(_physically_down):
        _keyboard_release_key(k)


_SPEED_TAG_RE = re.compile(
    r"^<x\s*([0-9]+\.?[0-9]*|[0-9]*\.[0-9]+)\s*>$",
    re.IGNORECASE,
)
_P_BLOCK_RE = re.compile(r"<p\s*>(.*?)</p\s*>", re.IGNORECASE | re.DOTALL)


def _parse_speed_tag(token: str) -> SheetSpeedMul | None:
    m = _SPEED_TAG_RE.match(token.strip())
    if not m:
        return None
    factor = float(m.group(1))
    if factor <= 0:
        raise ValueError(f"Speed factor in {token!r} must be positive")
    return SheetSpeedMul(factor=factor)


def _parse_tokens_to_steps(
    tokens: list[str],
    name_to_index: dict[str, int],
    tie_char: str,
) -> list[SheetItem]:
    steps: list[SheetItem] = []
    for token in tokens:
        sm = _parse_speed_tag(token)
        if sm is not None:
            steps.append(sm)
            continue
        if _is_tie_run(token, tie_char):
            steps.extend(SheetTieStep() for _ in range(len(token)))
            continue
        if token not in name_to_index:
            known = ", ".join(sorted(name_to_index, key=lambda s: (len(s), s)))
            raise ValueError(f"Unknown sheet token {token!r}. Known names: {known}")
        steps.append(SheetKeyStep(name_to_index[token]))
    return steps


def _parse_plain_sheet_chunk(
    chunk: str,
    name_to_index: dict[str, int],
    tie_char: str,
) -> list[SheetItem]:
    steps: list[SheetItem] = []
    for raw_line in chunk.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("/"):
            continue
        steps.extend(_parse_tokens_to_steps(line.split(), name_to_index, tie_char))
    return steps


def _parse_parallel_inner(
    inner: str,
    name_to_index: dict[str, int],
    tie_char: str,
) -> SheetParallel:
    normalized = " ".join(inner.split())
    parts = [p.strip() for p in normalized.split("/") if p.strip()]
    if not parts:
        raise ValueError(
            "<p> block must contain at least one voice (non-empty segment)."
        )
    voices: list[tuple[SheetItem, ...]] = []
    for p in parts:
        voices.append(tuple(_parse_tokens_to_steps(p.split(), name_to_index, tie_char)))
    return SheetParallel(voices=tuple(voices))


def _parse_sheet(text: str, keys: list[Note], tie_char: str) -> list[SheetItem]:
    """Parse sheet: ``<p>…</p>`` parallel blocks, ``<xN>`` tags, notes."""
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


def _play_parallel(
    voices: tuple[tuple[SheetItem, ...], ...],
    speed_mul: float,
) -> float:
    """Play aligned columns: each voice advances one step per column, same hold for all."""
    V = len(voices)
    max_len = max((len(voices[v]) for v in range(V)), default=0)
    voice_held: list[str | None] = [None] * V

    def other_holds_key(key: str, except_v: int) -> bool:
        for w, hk in enumerate(voice_held):
            if w != except_v and hk == key:
                return True
        return False

    def release_voice(v: int) -> None:
        k = voice_held[v]
        if k is None:
            return
        voice_held[v] = None
        if not other_holds_key(k, v):
            _keyboard_release_key(k)

    def voice_holds_key_full_column(v: int, col: int) -> bool:
        it = voices[v][col]
        if isinstance(it, SheetSpeedMul):
            return False
        nxt = voices[v][col + 1] if col + 1 < len(voices[v]) else None
        if isinstance(it, SheetTieStep):
            return nxt is not None and isinstance(nxt, SheetTieStep)
        if isinstance(it, SheetKeyStep):
            ki = KEYS[it.key_index]
            return (
                ki.key is not None and nxt is not None and isinstance(nxt, SheetTieStep)
            )
        return False

    def voice_should_release_at_column_boundary(v: int, col: int) -> bool:
        it = voices[v][col]
        if isinstance(it, SheetSpeedMul):
            return False
        nxt = voices[v][col + 1] if col + 1 < len(voices[v]) else None
        if isinstance(it, SheetTieStep):
            return nxt is None or not isinstance(nxt, SheetTieStep)
        assert isinstance(it, SheetKeyStep)
        ki = KEYS[it.key_index]
        if ki.key is not None and nxt is not None and isinstance(nxt, SheetTieStep):
            return False
        return True

    for col in range(max_len):
        if pause_event.is_set() or shutdown_event.is_set():
            break

        for v in range(V):
            if col < len(voices[v]):
                it = voices[v][col]
                if isinstance(it, SheetSpeedMul):
                    speed_mul = it.factor

        hold_sec = _effective_hold_seconds(speed_mul)
        press_part, tail = _beat_press_and_tail(hold_sec, speed_mul)

        for v in range(V):
            if col >= len(voices[v]):
                continue
            it = voices[v][col]
            if isinstance(it, SheetSpeedMul):
                continue
            if isinstance(it, SheetTieStep):
                continue
            if isinstance(it, SheetKeyStep):
                ki = KEYS[it.key_index]
                if ki.key is not None:
                    if voice_held[v] is not None and voice_held[v] != ki.key:
                        release_voice(v)
                    voice_held[v] = ki.key
                    _keyboard_press_key(ki.key)
                else:
                    release_voice(v)

        if not wait_or_pause(press_part):
            break

        for v in range(V):
            if col >= len(voices[v]):
                continue
            it = voices[v][col]
            if isinstance(it, SheetSpeedMul):
                continue
            if voice_holds_key_full_column(v, col):
                continue
            if voice_should_release_at_column_boundary(v, col):
                release_voice(v)

        if tail > 0 and not wait_or_pause(tail):
            break

    for v in range(V):
        release_voice(v)
    return speed_mul


def main() -> None:
    n_keys = sum(1 for x in sheet_steps if isinstance(x, SheetKeyStep))
    n_ties = sum(1 for x in sheet_steps if isinstance(x, SheetTieStep))
    n_cmds = sum(1 for x in sheet_steps if isinstance(x, SheetSpeedMul))
    n_par = sum(1 for x in sheet_steps if isinstance(x, SheetParallel))
    print("Auto key hold tapper ready (start hotkey toggles pause/resume).")
    print(
        f"Sheet: {len(sheet_steps)} top-level items ({n_keys} notes, {n_ties} ties, "
        f"{n_cmds} speed steps, {n_par} <p> blocks)"
    )
    print(
        f"Tie character in sheet: {TIE_CHAR!r} (runs like {TIE_CHAR * 3} = 3 tie steps)"
    )
    print(f"Key definitions: {KEYS}")
    print(
        f"Beat: {single_beat_seconds}s (÷N via <xN>); "
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

            speed_mul = 1.0
            for i, item in enumerate(sheet_steps):
                step = i + 1
                if pause_event.is_set() or shutdown_event.is_set():
                    break

                if isinstance(item, SheetSpeedMul):
                    speed_mul = item.factor
                    h = _effective_hold_seconds(speed_mul)
                    print(
                        f"[{step}] speed x{speed_mul} -> hold={h}s (base/{speed_mul})"
                    )
                    continue

                if isinstance(item, SheetParallel):
                    cols = max((len(v) for v in item.voices), default=0)
                    print(
                        f"[{step}] <p> parallel: {len(item.voices)} voices, {cols} columns"
                    )
                    speed_mul = _play_parallel(item.voices, speed_mul)
                    continue

                hold_sec = _effective_hold_seconds(speed_mul)
                press_part, tail = _beat_press_and_tail(hold_sec, speed_mul)
                nxt = _next_item_after(i)

                if isinstance(item, SheetTieStep):
                    # Tie: extend previous key for one beat; early release only when tie ends.
                    print(f"[{step}] -")
                    tie_continues = nxt is not None and isinstance(nxt, SheetTieStep)
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

                idx = item.key_index
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

                continues_into_tie = ki.key is not None and isinstance(
                    nxt, SheetTieStep
                )
                if continues_into_tie:
                    if not wait_or_pause(hold_sec):
                        break
                else:
                    if not wait_or_pause(press_part):
                        break
                    release_current_if_any()
                    if tail > 0 and not wait_or_pause(tail):
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
