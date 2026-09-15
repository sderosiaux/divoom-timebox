"""16x16 framebuffer, colour parsing, and image/text sources."""

from PIL import Image, ImageDraw, ImageFont

from .proto import MAX_COLORS, RGB, SCREEN

NAMED: dict[str, RGB] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 128, 0),
    "purple": (128, 0, 255),
    "pink": (255, 96, 160),
    "gray": (128, 128, 128),
}


def parse_color(text: str) -> RGB:
    key = text.strip().lower()
    if key in NAMED:
        return NAMED[key]
    hexpart = key.lstrip("#")
    if len(hexpart) == 3:
        hexpart = "".join(c * 2 for c in hexpart)
    if len(hexpart) != 6:
        raise ValueError(f"unknown colour {text!r}; try a name or #rrggbb")
    return tuple(int(hexpart[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


class Canvas:
    def __init__(self) -> None:
        self.pixels: list[RGB] = [(0, 0, 0)] * (SCREEN * SCREEN)

    def fill(self, color: RGB) -> None:
        self.pixels = [color] * (SCREEN * SCREEN)

    def set(self, x: int, y: int, color: RGB) -> None:
        if not (0 <= x < SCREEN and 0 <= y < SCREEN):
            raise ValueError(f"({x},{y}) is off screen")
        self.pixels[x + y * SCREEN] = color

    def checker(self, a: RGB, b: RGB, size: int = 2) -> None:
        self.pixels = [
            a if ((x // size) + (y // size)) % 2 == 0 else b
            for y in range(SCREEN)
            for x in range(SCREEN)
        ]

    def palette_probe(self, size: int) -> None:
        """`size` distinct colours laid out as seven wide bands.

        Palette size drives bits-per-pixel, so this isolates the wire format. A
        probe that cycled colours per pixel would instead measure how well the
        camera grid lines up, since a period-3 cycle lands diagonally on a
        16-wide screen while a period-4 one lands in clean columns. Wide bands
        make misalignment cost a handful of cells at six boundaries, no more.
        """
        hues = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
        bands = min(len(hues), size)

        colors: list[RGB] = []
        for j in range(size):
            band = j * bands // size
            start = -(-band * size // bands)
            level = 255 - (j - start) * 3
            colors.append(tuple(c * level for c in hues[band]))  # type: ignore[arg-type]

        self.pixels = [colors[k * size // (SCREEN * SCREEN)] for k in range(SCREEN * SCREEN)]

    def gradient(self) -> None:
        self.pixels = [
            (x * 17, y * 17, 255 - x * 17)
            for y in range(SCREEN)
            for x in range(SCREEN)
        ]

    def border(self, color: RGB) -> None:
        for i in range(SCREEN):
            self.set(i, 0, color)
            self.set(i, SCREEN - 1, color)
            self.set(0, i, color)
            self.set(SCREEN - 1, i, color)

    def preview(self) -> str:
        lines = []
        for y in range(SCREEN):
            row = []
            for x in range(SCREEN):
                r, g, b = self.pixels[x + y * SCREEN]
                row.append(f"\x1b[38;2;{r};{g};{b}m██")
            lines.append("".join(row) + "\x1b[0m")
        return "\n".join(lines)


# Measured against the panel: a full-white frame photographs strongly blue, and
# only comes out neutral when sent at roughly (255, 210, 165). Off by default —
# faithful output first, warmth on request.
WARM = (1.0, 0.82, 0.65)


def white_balance(pixels: list[RGB], amount: float) -> list[RGB]:
    """Pull the cold cast out of the panel; 0 sends what you asked for, 1 corrects fully."""
    if amount <= 0:
        return pixels
    gain = [1 - (1 - w) * min(amount, 1.0) for w in WARM]
    return [tuple(int(v * g) for v, g in zip(px, gain)) for px in pixels]  # type: ignore[misc]


def reduce_palette(pixels: list[RGB], limit: int = MAX_COLORS) -> list[RGB]:
    """Merge near-identical colours until the frame fits the device's palette."""
    if len({*pixels}) <= limit:
        return pixels
    img = Image.new("RGB", (SCREEN, SCREEN))
    img.putdata(pixels)
    return list(img.convert("P", palette=Image.Palette.ADAPTIVE, colors=limit).convert("RGB").getdata())  # type: ignore[arg-type]


def _flatten(frame: Image.Image) -> list[RGB]:
    rgba = frame.convert("RGBA")
    if rgba.size != (SCREEN, SCREEN):
        rgba = rgba.resize((SCREEN, SCREEN), Image.Resampling.NEAREST)
    out = []
    for y in range(SCREEN):
        for x in range(SCREEN):
            r, g, b, a = rgba.getpixel((x, y))  # type: ignore[misc]
            out.append((r, g, b) if a > 32 else (0, 0, 0))
    return reduce_palette(out)


def load_image(path: str) -> list[tuple[list[RGB], int]]:
    """Returns (pixels, duration_ms) per frame; a still image gives one frame."""
    frames = []
    with Image.open(path) as img:
        try:
            while True:
                composed = Image.new("RGBA", img.size)
                composed.paste(img, (0, 0), img.convert("RGBA"))
                frames.append((_flatten(composed), int(img.info.get("duration", 100))))
                img.seek(img.tell() + 1)
        except EOFError:
            pass
    return frames


def render_text(text: str, color: RGB, background: RGB = (0, 0, 0),
                duration_ms: int = 80, size: int = 14) -> list[tuple[list[RGB], int]]:
    """Scroll `text` right-to-left.

    `fontmode = "1"` turns off antialiasing, and it is not cosmetic. Smoothed
    edges put 60+ near-identical greys in every frame, which pushes the palette
    to 7 bits per pixel and the whole animation past what the device will
    accept — it gives up and falls back to the clock. At 16 pixels tall there is
    nothing to smooth anyway.
    """
    font = ImageFont.load_default(size)

    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    probe.fontmode = "1"
    width = int(probe.textbbox((0, 0), text, font=font)[2])

    total = SCREEN + width + SCREEN
    strip = Image.new("RGB", (total, SCREEN), background)
    draw = ImageDraw.Draw(strip)
    draw.fontmode = "1"
    draw.text((SCREEN, (SCREEN - size) // 2), text, font=font, fill=color)

    step = max(1, -(-(total - SCREEN) // 58))
    return [
        (_flatten(strip.crop((offset, 0, offset + SCREEN, SCREEN))), duration_ms)
        for offset in range(0, total - SCREEN, step)
    ]
