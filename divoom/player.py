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


def play(link: Link, effect: Effect, fps: float = 20, seconds: float | None = None,
         colors: int = 255, warm: float = 0.0) -> Played:
    """Run `effect` until `seconds` elapse, or forever if None.

    Frames are dropped rather than queued when generation falls behind, so the
    animation stays in step with the clock instead of drifting into slow motion.
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

        pixels = reduce_palette(white_balance(effect.next(), warm), colors)
        link.send(*proto.image_messages(pixels))
        shown += 1

    return Played(shown, time.monotonic() - start)
