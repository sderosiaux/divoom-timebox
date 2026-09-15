"""Stream effect frames to the panel on a wall clock."""

import itertools
import time
from dataclasses import dataclass

from . import proto
from .bridge import Link
from .canvas import reduce_palette, white_balance
from .effects import Effect


@dataclass
class Played:
    frames: int
    seconds: float

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds else 0.0


GAMMA = 2.2


def dim(pixels, level: float):
    """Scale light output, not stored values.

    Multiplying sRGB numbers directly looks wrong on the way down: the last
    third of the fade collapses in a couple of frames. Light adds linearly, so
    the multiplier belongs in linear space.
    """
    if level >= 1.0:
        return pixels
    if level <= 0.0:
        return [(0, 0, 0)] * len(pixels)
    factor = level ** (1 / GAMMA)
    return [(int(r * factor), int(g * factor), int(b * factor)) for r, g, b in pixels]


def play(link: Link, effect: Effect, fps: float = 20, seconds: float | None = None,
         colors: int = 255, warm: float = 0.0, fade: float = 0.0) -> Played:
    """Run `effect` until `seconds` elapse, or forever if None.

    Frames are dropped rather than queued when generation falls behind, so the
    animation stays in step with the clock instead of drifting into slow motion.

    `fade` seconds of dip at each end lets a caller cut between effects without
    the jump that swapping frames outright produces.
    """
    period = 1 / fps
    start = time.monotonic()
    shown = 0

    for index in itertools.count():
        deadline = start + index * period
        now = time.monotonic()
        if seconds is not None and now - start >= seconds:
            break
        if deadline < now - period:
            effect.next()
            continue
        time.sleep(max(0.0, deadline - now))

        level = 1.0
        if fade > 0:
            elapsed = now - start
            level = min(elapsed / fade, 1.0)
            if seconds is not None:
                level = min(level, (seconds - elapsed) / fade)
            level = max(0.0, min(1.0, level))

        pixels = reduce_palette(white_balance(dim(effect.next(), level), warm), colors)
        link.send(*proto.image_messages(pixels))
        shown += 1

    return Played(shown, time.monotonic() - start)
