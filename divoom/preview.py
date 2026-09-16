"""Render the actual effect frames as an LED contact sheet without Bluetooth.

    python -m divoom.preview
    python -m divoom.preview snake hyperspace --output /tmp/space.gif
"""

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import effects
from .proto import SCREEN


def sheet(names: list[str], frames: list[list], scale: int = 8) -> Image.Image:
    columns = min(5, len(names))
    tile = SCREEN * scale
    width, height = tile + 24, tile + 44
    image = Image.new("RGB", (columns * width, math.ceil(len(names) / columns) * height), (12, 12, 16))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(14)
    for index, (name, pixels) in enumerate(zip(names, frames)):
        left, top = index % columns * width + 12, index // columns * height + 10
        draw.text((left, top), name, font=font, fill=(225, 225, 230))
        top += 24
        for y in range(SCREEN):
            for x in range(SCREEN):
                color = pixels[y * SCREEN + x]
                draw.rectangle((left + x * scale, top + y * scale,
                                left + (x + 1) * scale - 2, top + (y + 1) * scale - 2), fill=color)
    return image


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("effects", nargs="*", help="effect names; defaults to the arcade collection")
    ap.add_argument("--output", type=Path, default=Path("arcade.gif"))
    ap.add_argument("--seconds", type=float, default=12)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    names = args.effects or list(effects.ARCADE)
    if any(name not in effects.ABSTRACT for name in names):
        ap.error(f"choose from {', '.join(sorted(effects.ABSTRACT))}")
    if not math.isfinite(args.seconds) or not 0.1 <= args.seconds <= 60:
        ap.error("--seconds must be between 0.1 and 60")
    if args.output.suffix.lower() != ".gif":
        ap.error("--output must end in .gif (a PNG is also saved alongside it)")
    generators = [effects.build(name, args.seed) for name in names]
    images = []
    for tick in range(round(args.seconds * 20)):
        frames = [effect.next() for effect in generators]
        if tick % 2 == 0:  # simulate 20 fps, export at 10 fps
            images.append(sheet(names, frames))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    images[len(images) // 2].save(args.output.with_suffix(".png"))
    images[0].save(args.output, save_all=True, append_images=images[1:], duration=100, loop=0)
    print(f"{args.output} ({len(images)} frames) + {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
