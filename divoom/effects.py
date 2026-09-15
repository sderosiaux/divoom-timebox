"""Frame generators for the 16x16 panel.

Every effect is stepped rather than sampled at a time `t`: some of them (fire,
life) carry state that cannot be recomputed from a clock. Seeding makes frame N
reproducible anyway, which is what the camera tests rely on.

Frames are streamed as still images instead of uploaded as device animations —
the Evo caps an animation near 60 frames, and the link sustains 30fps, so
driving it from here removes the limit entirely.
"""

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

from .proto import RGB, SCREEN

N = SCREEN


def _as_pixels(frame: np.ndarray) -> list[RGB]:
    flat = np.clip(frame, 0, 255).astype(np.uint8).reshape(-1, 3)
    return [(int(r), int(g), int(b)) for r, g, b in flat]


def _hsv(h: np.ndarray, s: np.ndarray | float, v: np.ndarray | float) -> np.ndarray:
    """Vectorised HSV→RGB; h wraps, s and v in [0, 1]."""
    h = np.asarray(h) % 1.0
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    ones = np.ones_like(h)
    r = np.choose(i, [v * ones, q * ones, p * ones, p * ones, t * ones, v * ones])
    g = np.choose(i, [t * ones, v * ones, v * ones, q * ones, p * ones, p * ones])
    b = np.choose(i, [p * ones, p * ones, t * ones, v * ones, v * ones, q * ones])
    return np.stack([r, g, b], axis=-1) * 255


class Effect:
    name = "effect"

    def next(self) -> list[RGB]:
        raise NotImplementedError


class Plasma(Effect):
    """Interfering sine fields — the archetypal demoscene screensaver."""

    name = "plasma"

    def __init__(self, seed: int = 0, speed: float = 0.07):
        self.speed = speed
        self.t = seed * 1.7
        ys, xs = np.mgrid[0:N, 0:N]
        self.x, self.y = xs / N, ys / N

    def next(self) -> list[RGB]:
        t, self.t = self.t, self.t + self.speed
        field = (
            np.sin(self.x * 6 + t)
            + np.sin(self.y * 6 + t * 1.3)
            + np.sin((self.x + self.y) * 5 + t * 0.7)
            + np.sin(np.hypot(self.x - 0.5, self.y - 0.5) * 14 - t * 1.1)
        )
        return _as_pixels(_hsv(field / 8 + 0.5, 1.0, 1.0))


class Fire(Effect):
    """Heat seeded along the bottom row, cooled and drifted upward."""

    name = "fire"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.heat = np.zeros((N + 1, N))
        ramp = np.linspace(0, 1, 256)
        self.palette = _hsv(ramp * 0.12, np.clip(2.4 - ramp * 2.4, 0, 1), np.clip(ramp * 2.2, 0, 1))

    def next(self) -> list[RGB]:
        self.heat[-1] = self.rng.random(N) ** 0.5
        nxt = self.heat.copy()
        for y in range(N):
            below = self.heat[y + 1]
            spread = (below + np.roll(below, 1) + np.roll(below, -1)) / 3
            nxt[y] = np.clip(spread - self.rng.random(N) * 0.12, 0, 1)
        self.heat = nxt
        return _as_pixels(self.palette[(self.heat[:N] * 255).astype(int)])


class Life(Effect):
    """Conway on a torus; reseeds when it stalls so it never sits still."""

    name = "life"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.hue = self.rng.random()
        self._seed_board()
        self.history: list[bytes] = []

    def _seed_board(self) -> None:
        self.board = np.array(
            [[1 if self.rng.random() < 0.35 else 0 for _ in range(N)] for _ in range(N)]
        )
        self.age = np.zeros((N, N))

    def next(self) -> list[RGB]:
        b = self.board
        neighbours = sum(
            np.roll(np.roll(b, dy, 0), dx, 1)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dy, dx) != (0, 0)
        )
        self.board = ((neighbours == 3) | ((b == 1) & (neighbours == 2))).astype(int)
        self.age = np.where(self.board == 1, self.age + 1, 0)

        key = self.board.tobytes()
        self.history.append(key)
        if key in self.history[:-1] or self.board.sum() == 0:
            self.hue = (self.hue + 0.27) % 1.0
            self._seed_board()
        self.history = self.history[-12:]

        value = np.clip(self.age / 6, 0, 1) * 0.55 + 0.45
        frame = _hsv(np.full((N, N), self.hue) + self.age * 0.01, 0.75, value)
        return _as_pixels(frame * self.board[..., None])


class Rain(Effect):
    """Falling columns with fading trails."""

    name = "rain"

    def __init__(self, seed: int = 0, hue: float = 0.33):
        self.rng = random.Random(seed)
        self.hue = hue
        self.heads = [self.rng.uniform(-N, 0) for _ in range(N)]
        self.speeds = [self.rng.uniform(0.25, 0.9) for _ in range(N)]

    def next(self) -> list[RGB]:
        value = np.zeros((N, N))
        for x in range(N):
            self.heads[x] += self.speeds[x]
            if self.heads[x] - 8 > N:
                self.heads[x] = self.rng.uniform(-8, -1)
                self.speeds[x] = self.rng.uniform(0.25, 0.9)
            head = self.heads[x]
            for y in range(N):
                behind = head - y
                if 0 <= behind < 8:
                    value[y, x] = 1 - behind / 8
        return _as_pixels(_hsv(np.full((N, N), self.hue), 1 - value ** 3, value))


class Starfield(Effect):
    """Stars rushing past, brighter as they get closer."""

    name = "stars"

    def __init__(self, seed: int = 0, count: int = 40):
        self.rng = random.Random(seed)
        self.stars = [self._spawn() for _ in range(count)]

    def _spawn(self) -> list[float]:
        return [self.rng.uniform(-1, 1), self.rng.uniform(-1, 1), self.rng.uniform(0.15, 1.0)]

    def next(self) -> list[RGB]:
        frame = np.zeros((N, N, 3))
        for star in self.stars:
            star[2] -= 0.022
            if star[2] <= 0.05:
                star[:] = self._spawn()
                star[2] = 1.0
            x = int(star[0] / star[2] * (N / 4) + N / 2)
            y = int(star[1] / star[2] * (N / 4) + N / 2)
            if 0 <= x < N and 0 <= y < N:
                level = min(255, (1 - star[2]) * 340)
                frame[y, x] = np.maximum(frame[y, x], level)
        return _as_pixels(frame)


class Bounce(Effect):
    """The DVD logo, in spirit: a block that changes colour on every wall hit."""

    name = "bounce"

    def __init__(self, seed: int = 0, size: int = 4):
        self.rng = random.Random(seed)
        self.size = size
        self.x, self.y = self.rng.uniform(0, N - size), self.rng.uniform(0, N - size)
        self.dx, self.dy = self.rng.choice([-0.4, 0.4]), self.rng.choice([-0.3, 0.3])
        self.hue = self.rng.random()

    def next(self) -> list[RGB]:
        self.x += self.dx
        self.y += self.dy
        limit = N - self.size
        for axis in ("x", "y"):
            pos, vel = getattr(self, axis), getattr(self, "d" + axis)
            if pos < 0 or pos > limit:
                setattr(self, axis, min(max(pos, 0), limit))
                setattr(self, "d" + axis, -vel)
                self.hue = (self.hue + 0.17) % 1.0

        frame = np.zeros((N, N, 3))
        color = _hsv(np.array([self.hue]), 1.0, 1.0)[0]
        x0, y0 = int(self.x), int(self.y)
        frame[y0:y0 + self.size, x0:x0 + self.size] = color
        return _as_pixels(frame)


class Rings(Effect):
    """Concentric pulses travelling out from a wandering centre."""

    name = "rings"

    def __init__(self, seed: int = 0):
        self.t = seed * 3.0
        ys, xs = np.mgrid[0:N, 0:N]
        self.xs, self.ys = xs, ys

    def next(self) -> list[RGB]:
        t, self.t = self.t, self.t + 0.12
        cx = N / 2 + math.sin(t * 0.21) * 4
        cy = N / 2 + math.cos(t * 0.17) * 4
        d = np.hypot(self.xs - cx, self.ys - cy)
        wave = (np.sin(d * 1.1 - t * 1.6) + 1) / 2
        return _as_pixels(_hsv(d * 0.035 + t * 0.02, 0.9, wave ** 2))


class Pan(Effect):
    """Ken Burns over an image far larger than the panel: drift plus slow zoom."""

    name = "pan"

    def __init__(self, path: str | Path, seed: int = 0, speed: float = 0.35):
        self.source = Image.open(path).convert("RGB")
        if min(self.source.size) < N * 2:
            scale = (N * 3) / min(self.source.size)
            self.source = self.source.resize(
                (int(self.source.width * scale), int(self.source.height * scale)), Image.Resampling.LANCZOS
            )
        self.t = seed * 10.0
        self.speed = speed

    def next(self) -> list[RGB]:
        t, self.t = self.t, self.t + self.speed
        w, h = self.source.size
        shortest = min(w, h)

        # Breathe between a tight crop and most of the frame, drifting meanwhile.
        zoom = 0.30 + 0.35 * (math.sin(t * 0.012) + 1) / 2
        window = max(N, int(shortest * zoom))
        span_x, span_y = max(w - window, 1), max(h - window, 1)
        x = (math.sin(t * 0.008) + 1) / 2 * span_x
        y = (math.sin(t * 0.0061 + 1.3) + 1) / 2 * span_y

        crop = self.source.crop((int(x), int(y), int(x) + window, int(y) + window))
        return list(crop.resize((N, N), Image.Resampling.LANCZOS).getdata())  # type: ignore[arg-type]


class Logo(Effect):
    """A mark kept alive, several ways.

    Three things were wrong before and matter more than any of the motion. The
    source PNG carries a wide margin, so crop to the artwork or the mark sits
    tiny in the middle. The mark is rendered by coverage rather than resampled,
    because smoothing a geometric logo to 16 pixels turns it into a smudge. And
    the window must divide evenly into 16 cells — at 6.375 pixels per cell the
    threshold tipped differently on each side and one half came out fatter.

    Every mode holds the logo's own four-fold symmetry: the coverage field is
    averaged over its four quarter-turns before thresholding, so no arm can end
    up heavier than another at any angle.
    """

    name = "logo"
    MODES = ("pulse", "spin", "zoom", "ripple", "orbit", "beat")

    def __init__(self, path: str | Path, mode: str = "pulse", seed: int = 0,
                 fps: float = 20, coverage: float = 0.62, bleed: int = 1,
                 breath: float = 3.0, pulse: float = 6.0,
                 spin_seconds: float = 6.0, zoom_seconds: float = 5.0,
                 softness: float = 0.45, blur: int = 6, supersample: int = 16):
        if mode not in self.MODES:
            raise ValueError(f"unknown mode {mode!r}; try {', '.join(self.MODES)}")
        self.mode = mode
        self.name = f"logo:{mode}"
        self.fps = fps
        self.coverage = coverage
        self.breath = breath
        self.pulse = pulse
        self.spin_seconds = spin_seconds
        self.zoom_seconds = zoom_seconds
        self.softness = softness
        self.blur = blur
        # Coverage is measured by supersampling: render large, average down.
        # Finer sampling buys smoother edges and, since every angle is cached,
        # costs nothing after the first turn.
        self.supersample = supersample
        self.frame = seed * 13

        source = np.asarray(Image.open(path).convert("RGB"), dtype=float)
        edges = np.concatenate([source[0, :], source[-1, :], source[:, 0], source[:, -1]])
        self.background = edges.mean(axis=0)

        ink = np.abs(source - self.background).sum(axis=2)
        ys, xs = np.nonzero(ink > ink.max() * 0.35)
        if len(xs):
            top, bottom = ys.min(), ys.max() + 1
            left, right = xs.min(), xs.max() + 1
            centre_y, centre_x = (top + bottom) / 2, (left + right) / 2
            half = max(centre_y - top, bottom - centre_y,
                       centre_x - left, right - centre_x) + bleed
            side = 2 * int(round(half))

            margin = side
            padded = np.pad(source, ((margin, margin), (margin, margin), (0, 0)))
            padded[:margin] = padded[-margin:] = self.background
            padded[:, :margin] = padded[:, -margin:] = self.background
            source = padded[int(round(centre_y - side / 2)) + margin:][:side,
                     int(round(centre_x - side / 2)) + margin:][:, :side]

        self.square = Image.fromarray(source.astype("uint8"))
        self.fill = tuple(int(v) for v in self.background)
        # Motion blur asks for several sub-angles per frame, and `orbit` varies
        # angle and scale together, so the cache grows without bound on a mode
        # meant to run for hours. Evict oldest-first once it is large enough to
        # cover a full cycle.
        self._cache: dict[tuple[float, float], np.ndarray] = {}
        self._cache_limit = 4096

        core = self._alpha() > 0.9
        sampled = np.asarray(self.square.resize((N, N), Image.BOX), dtype=float)[core]
        self.tone = sampled.mean(axis=0) if len(sampled) else np.array([138.0, 193.0, 85.0])

        ys, xs = np.mgrid[0:N, 0:N]
        self.radius = np.hypot(xs - (N - 1) / 2, ys - (N - 1) / 2) / (N / 2)

    def _alpha(self, angle: float = 0.0, scale: float = 1.0) -> np.ndarray:
        """Coverage per LED, 0 to 1, rather than a yes/no mask.

        A hard threshold makes the mark jump a whole pixel at a time, which on a
        16-wide panel reads as stepping rather than turning. Letting partial
        coverage light an LED partially buys sub-pixel motion: the edge fades
        across as the shape passes. Only the boundary softens — the core stays
        solid, which is what smoothing the whole image got wrong.
        """
        key = (round(angle % 90, 1), round(scale, 2))
        if key in self._cache:
            return self._cache[key]

        image = self.square
        if key[1] != 1.0:
            inner = max(8, int(round(image.width * key[1])))
            shrunk = image.resize((inner, inner), Image.LANCZOS)
            canvas = Image.new("RGB", image.size, self.fill)
            canvas.paste(shrunk, ((image.width - inner) // 2, (image.height - inner) // 2))
            image = canvas
        if key[0]:
            image = image.rotate(key[0], resample=Image.BICUBIC, fillcolor=self.fill)

        # Through a multiple of the panel size: straight to 16 leaves a
        # fractional number of source pixels per cell.
        big = image.resize((N * self.supersample,) * 2, Image.LANCZOS)
        field = np.abs(np.asarray(big.resize((N, N), Image.BOX), dtype=float)
                       - self.background).sum(axis=2)
        field = sum(np.rot90(field, k) for k in range(4)) / 4

        peak = field.max() or 1.0
        edge = self.coverage * self.softness
        low, high = peak * max(self.coverage - edge, 0.02), peak * (self.coverage + edge)
        ramp = np.clip((field - low) / max(high - low, 1e-6), 0, 1)
        if len(self._cache) >= self._cache_limit:
            del self._cache[next(iter(self._cache))]
        self._cache[key] = (ramp * ramp * (3 - 2 * ramp)).astype(np.float32)  # smoothstep
        return self._cache[key]

    def _smear(self, angle: float, scale: float, span: float) -> np.ndarray:
        """Average the coverage across the angle swept during one frame.

        Proper motion blur rather than a softened still: the trailing edge of
        each arm dims while the leading edge brightens, which is what sells a
        rotation that only advances a fraction of a pixel per frame.
        """
        if self.blur < 2 or span <= 0:
            return self._alpha(angle, scale)
        steps = [angle - span * i / (self.blur - 1) for i in range(self.blur)]
        return sum(self._alpha(a, scale) for a in steps) / self.blur

    @staticmethod
    def _ease(x: float) -> float:
        return (1 - math.cos(math.pi * x)) / 2

    def next(self) -> list[RGB]:
        now, self.frame = self.frame / self.fps, self.frame + 1

        angle, scale = 0.0, 1.0
        breathe = 0.78 + 0.22 * (math.sin(2 * math.pi * now / self.breath) + 1) / 2
        glow = None

        if self.mode == "pulse":
            phase = (now % self.pulse) / self.pulse
            if phase < 0.34:
                glow = np.exp(-(((self.radius - phase / 0.34 * 1.15) * 4) ** 2))

        elif self.mode == "spin":
            angle = 90 * (now % self.spin_seconds) / self.spin_seconds

        elif self.mode == "zoom":
            # In to a point and back out, easing at both ends so it breathes
            # rather than bounces.
            scale = 0.25 + 0.75 * self._ease((now % self.zoom_seconds) / self.zoom_seconds * 2 % 2)
            breathe = 1.0

        elif self.mode == "ripple":
            wave = np.sin(2 * math.pi * (self.radius * 1.6 - now / self.pulse * 2))
            glow = (wave + 1) / 2 * 0.9
            breathe = 0.62

        elif self.mode == "orbit":
            angle = 90 * (now % (self.spin_seconds * 2)) / (self.spin_seconds * 2)
            scale = 0.55 + 0.45 * self._ease((now % self.zoom_seconds) / self.zoom_seconds * 2 % 2)

        elif self.mode == "beat":
            # Two thumps then a rest, on the rhythm of a heart.
            beat = now % self.breath
            thump = max(math.exp(-((beat / 0.16) ** 2)),
                        0.75 * math.exp(-(((beat - 0.42) / 0.16) ** 2)))
            scale = 0.82 + 0.18 * thump
            breathe = 0.74 + 0.36 * thump

        tone = self.tone * breathe
        if glow is not None:
            tone = tone + glow[..., None] * 85

        span = 0.0
        if self.mode == "spin":
            span = 90 / (self.spin_seconds * self.fps)
        elif self.mode == "orbit":
            span = 90 / (self.spin_seconds * 2 * self.fps)
        alpha = self._smear(angle, scale, span)[..., None]

        lit = np.broadcast_to(tone, (N, N, 3)) if tone.ndim == 1 else tone
        frame = self.background * (1 - alpha) + lit * alpha
        return _as_pixels(frame)


ABSTRACT = {e.name: e for e in (Plasma, Fire, Life, Rain, Starfield, Bounce, Rings)}


def build(name: str, seed: int = 0) -> Effect:
    if name not in ABSTRACT:
        raise ValueError(f"unknown effect {name!r}; try {', '.join(sorted(ABSTRACT))}")
    return ABSTRACT[name](seed)  # type: ignore[call-arg]
