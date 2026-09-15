"""Keep the panel running unattended.

Meant to be owned by launchd rather than a terminal, so it outlives the session
that started it. The retry loop is the point: the speaker drops its Bluetooth
link when it has been idle, its battery runs out, and macOS occasionally grabs
it back as an audio device. None of that should need a human.

    python -m divoom.serve logo spin
    python -m divoom.serve screensaver --each 60
    python -m divoom.serve face doom
"""

import argparse
import itertools
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

from . import effects, faces, player, proto
from .bridge import BridgeError, Link, paired

LOGO = Path(__file__).resolve().parent.parent / "assets/logo/conduktor.png"

# Back off on failure, but never so far that a speaker coming back on charge
# waits minutes to be noticed.
FIRST_WAIT = 5.0
MAX_WAIT = 120.0
SETTLED = 60.0          # a run this long counts as healthy; reset the backoff


def log(message: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {message}", flush=True)


def segments(args) -> Iterator[tuple[object, float | None]]:
    """What to play, forever. Each item is an effect and how long to hold it."""
    if args.what == "logo":
        while True:
            yield effects.Logo(str(args.path or LOGO), args.mode, fps=args.fps), None

    elif args.what == "play":
        for seed in itertools.count(1):
            yield effects.build(args.mode, seed), None

    elif args.what == "face":
        for seed in itertools.count(1):
            yield faces.Face(seed, faces.CHARACTERS[args.mode], fps=args.fps,
                             pace=args.pace), None

    elif args.what == "screensaver":
        for seed in itertools.count(1):
            for name in sorted(effects.ABSTRACT):
                yield effects.build(name, seed), args.each

    elif args.what == "logocycle":
        for seed in itertools.count(1):
            for mode in effects.Logo.MODES:
                yield effects.Logo(str(args.path or LOGO), mode, seed, fps=args.fps), args.each


def device(hint: str | None) -> tuple[str, str]:
    def wanted(row: tuple[str, str, str]) -> bool:
        address, _, name = row
        if hint:
            return hint.lower() in address.lower() or hint.lower() in name.lower()
        return any(k in name.lower() for k in ("divoom", "timebox", "evo"))

    matches = [row for row in paired() if wanted(row)]
    if not matches:
        raise BridgeError("no Timebox among the paired devices")
    return matches[0][0], matches[0][2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="divoom.serve")
    ap.add_argument("what", choices=["logo", "logocycle", "play", "face", "screensaver"])
    ap.add_argument("mode", nargs="?", default="spin",
                    help="logo mode, effect name, or character")
    ap.add_argument("--path", help="image for the logo modes")
    ap.add_argument("--fps", type=float, default=20)
    ap.add_argument("--warm", type=float, default=0.3)
    ap.add_argument("--bright", type=int, default=85)
    ap.add_argument("--colors", type=int, default=255)
    ap.add_argument("--each", type=float, default=45, help="seconds per item when cycling")
    ap.add_argument("--pace", type=float, default=2.0, help="idle speed for faces")
    ap.add_argument("--channel", type=int, default=1)
    ap.add_argument("--mac", help="address or name fragment")
    args = ap.parse_args(argv)

    log(f"serving {args.what} {args.mode} at {args.fps:g} fps")
    wait = FIRST_WAIT

    while True:
        link = None
        started = time.monotonic()
        try:
            address, name = device(args.mac)
            link = Link(address, args.channel)
            log(f"connected to {name} ({address}), MTU {link.mtu}")

            link.send(proto.brightness(args.bright))
            time.sleep(0.3)
            # The Evo acknowledges the first image on a fresh channel and then
            # ignores it, so spend one frame before the real content.
            link.send(*proto.image_messages([(0, 0, 0)] * proto.PIXELS))
            time.sleep(0.4)

            wait = FIRST_WAIT
            for effect, seconds in segments(args):
                player.play(link, effect, args.fps, seconds, args.colors, args.warm)

        except KeyboardInterrupt:
            log("stopping")
            return 0
        except (BridgeError, OSError, ValueError) as exc:
            alive = time.monotonic() - started
            if alive > SETTLED:
                wait = FIRST_WAIT
            log(f"lost after {alive:.0f}s: {exc} — retrying in {wait:.0f}s")
            time.sleep(wait)
            wait = min(wait * 2, MAX_WAIT)
        finally:
            if link is not None:
                try:
                    link.close()
                except Exception:
                    pass


if __name__ == "__main__":
    sys.exit(main())
