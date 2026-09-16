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
from collections import deque
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


# Arcade artwork is drawn on the LED grid. Primary/secondary RGB colours keep
# neighbouring objects distinct without relying on subtle, unmeasured shades.
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
PINK = (255, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


def _dot(frame, x, y, color):
    x, y = round(x), round(y)
    if 0 <= x < N and 0 <= y < N:
        frame[y, x] = color


def _sprite(frame, x, y, rows, color):
    for dy, row in enumerate(rows):
        for dx, pixel in enumerate(row):
            if pixel != ".":
                _dot(frame, x + dx, y + dy, color)


def _path(start, targets, blocked, cells):
    """Shortest grid path; fixed neighbour order also makes replays reproducible."""
    queue, parents = deque([start]), {start: None}
    while queue:
        point = queue.popleft()
        if point in targets:
            route = []
            while point != start:
                route.append(point)
                point = parents[point]
            return route[::-1]
        x, y = point
        for nxt in ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)):
            if nxt in cells and nxt not in blocked and nxt not in parents:
                parents[nxt] = point
                queue.append(nxt)
    return []


class Snake(Effect):
    """Food-seeking snake, with a tail escape check before committing to a meal."""

    name = "snake"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.cells = {(x, y) for y in range(1, N - 1) for x in range(1, N - 1)}
        self.tick = self.pause = 0
        self._reset()

    def _reset(self):
        self.body = [(7, 8), (6, 8), (5, 8), (4, 8)]
        self._food()

    def _food(self):
        free = sorted(self.cells - set(self.body))
        self.food = self.rng.choice(free) if free else None

    def _move(self):
        route = _path(self.body[0], {self.food}, set(self.body[:-1]), self.cells)
        if route:
            future = self.body[:]
            for step in route:
                future.insert(0, step)
                if step != self.food:
                    future.pop()
            if not _path(future[0], {future[-1]}, set(future[:-1]), self.cells):
                route = []
        if not route:
            route = _path(self.body[0], {self.body[-1]}, set(self.body[:-1]), self.cells)
        if not route:
            self.pause = 20
            return
        head = route[0]
        self.body.insert(0, head)
        if head == self.food:
            self._food()
        else:
            self.body.pop()
        if len(self.body) >= 60 or self.food is None:
            self.pause = 20

    def next(self) -> list[RGB]:
        self.tick += 1
        if self.pause:
            self.pause -= 1
            if not self.pause:
                self._reset()
        elif self.tick % 3 == 0:
            self._move()
        frame = np.zeros((N, N, 3))
        frame[[0, -1], :] = frame[:, [0, -1]] = (0, 0, 48)
        for x, y in self.body:
            _dot(frame, x, y, GREEN)
        _dot(frame, *self.body[0], CYAN)
        if self.food:
            _dot(frame, *self.food, RED)
        return _as_pixels(frame)


class Pong(Effect):
    """Two opponents trade rebounds across the full panel."""

    name = "pong"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.tick = 0
        self.scores = [0, 0]
        self.paddles = [8.0, 8.0]
        self.targets = [8.0, 8.0]
        self._serve()

    def _serve(self):
        self.x, self.y = 7.5, self.rng.uniform(5, 11)
        self.dx = self.rng.choice((-0.36, 0.36))
        self.dy = self.rng.choice((-0.23, 0.23))
        self.pause = 16
        self.rally = 0
        self._aim()

    def _aim(self):
        side = int(self.dx > 0)
        contact = 14 if side else 1
        landing = self.y + self.dy * (contact - self.x) / self.dx
        # Reflect the prediction off the top/bottom walls. An aim is held for
        # the whole flight, rather than wobbling a paddle with a sine wave.
        landing = 15 - abs(landing % 30 - 15)
        error = self.rng.uniform(-0.5, 0.5)
        if self.rally > 3 and self.rng.random() < 0.3:
            error += self.rng.choice((-2.5, 2.5))
        self.targets[side] = max(1, min(14, landing + error))
        self.targets[1 - side] = 8

    def next(self) -> list[RGB]:
        self.tick += 1
        for side in range(2):
            self.paddles[side] += max(-0.34, min(0.34, self.targets[side] - self.paddles[side]))
        if self.pause:
            self.pause -= 1
        else:
            old_x = self.x
            self.x += self.dx
            self.y += self.dy
            if not 0 <= self.y <= 15:
                self.y = -self.y if self.y < 0 else 30 - self.y
                self.dy *= -1
            side = 0 if self.dx < 0 else 1
            crossed = old_x > 1 >= self.x if side == 0 else old_x < 14 <= self.x
            offset = self.y - round(self.paddles[side])
            if crossed and abs(offset) <= 1.5:
                # Only a crossing can hit. A missed ball must not get caught
                # retroactively by a paddle after passing behind its face.
                self.x = 2 - self.x if side == 0 else 28 - self.x
                self.dx = (-1 if side else 1) * min(0.58, abs(self.dx) + 0.018)
                self.dy = offset * 0.29
                if abs(self.dy) < 0.12:
                    self.dy = self.rng.choice((-0.12, 0.12))
                self.rally += 1
                self._aim()
            elif self.x < 0 or self.x > 15:
                self.scores[1 - side] += 1
                if max(self.scores) == 8:
                    self.scores = [0, 0]
                self._serve()
        frame = np.zeros((N, N, 3))
        frame[::2, 7] = (24, 24, 24)
        for x, y, color in ((0, self.paddles[0], CYAN), (15, self.paddles[1], PINK)):
            for offset in (-1, 0, 1):
                _dot(frame, x, y + offset, color)
        _dot(frame, self.x, self.y, WHITE)
        return _as_pixels(frame)


class Breakout(Effect):
    """A paddle clears four rows of bricks, then serves a fresh board."""

    name = "breakout"
    COLORS = (RED, YELLOW, GREEN, CYAN)

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self._reset()

    def _reset(self):
        self.bricks = {(x, y) for y in range(4) for x in range(4)}
        self._serve()

    def _serve(self):
        self.x, self.y = 7.0, 12.0
        self.dx, self.dy = self.rng.choice((-0.22, 0.22)), -0.32
        self.paddle = 7.0
        self.pause = 16

    def next(self) -> list[RGB]:
        if self.pause:
            self.pause -= 1
            if not self.pause and not self.bricks:
                self._reset()
        else:
            self.paddle = min(13, max(2, self.paddle + max(-0.3, min(0.3, self.x - self.paddle))))
            old_x, old_y = round(self.x), round(self.y)
            self.x += self.dx
            self.y += self.dy
            if not 0 <= self.x <= 15:
                self.x = max(0, min(15, self.x))
                self.dx *= -1
            if self.y < 0:
                self.y, self.dy = 0, abs(self.dy)
            x, y = round(self.x), round(self.y)
            brick = (x // 4, y // 2)
            if y % 2 == 1 and x % 4 < 3 and brick in self.bricks:
                self.bricks.remove(brick)
                if y != old_y:
                    self.dy *= -1
                elif x != old_x:
                    self.dx *= -1
                else:
                    self.dy *= -1
                if not self.bricks:
                    self.pause = 24
            if self.dy > 0 and self.y >= 13.5 and abs(self.x - self.paddle) <= 2.3:
                self.y, self.dy = 13.5, -abs(self.dy)
                self.dx = (self.x - self.paddle) * 0.16 + self.rng.uniform(-0.08, 0.08)
            elif self.y > 15:
                self._serve()
        frame = np.zeros((N, N, 3))
        for x, y in self.bricks:
            frame[1 + 2 * y, 4 * x:4 * x + 3] = self.COLORS[y]
        for dx in range(-2, 3):
            _dot(frame, self.paddle + dx, 15, CYAN)
        _dot(frame, self.x, self.y, WHITE)
        return _as_pixels(frame)


class Tetris(Effect):
    """Seven-bag falling blocks; an autoplayer favours lines and avoids holes."""

    name = "tetris"
    WIDTH, HEIGHT = 10, 15
    SHAPES = ("####", "##/##", ".#./###", ".##/##.", "##./.##", "#../###", "..#/###")
    COLORS = (BLACK, CYAN, YELLOW, PINK, GREEN, RED, BLUE, (255, 128, 0))

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.tick = self.pause = self.lines = 0
        self.bag = []
        self.clearing = []
        self.board = np.zeros((self.HEIGHT, self.WIDTH), dtype=int)
        self._spawn()

    def _fits(self, shape, x, y):
        return all(0 <= x + dx < self.WIDTH and 0 <= y + dy < self.HEIGHT
                   and self.board[y + dy, x + dx] == 0 for dx, dy in shape)

    @staticmethod
    def _rotate(shape):
        turned = [(-y, x) for x, y in shape]
        mx, my = min(x for x, y in turned), min(y for x, y in turned)
        return sorted((x - mx, y - my) for x, y in turned)

    def _spawn(self):
        if not self.bag:
            self.bag = list(range(7))
            self.rng.shuffle(self.bag)
        kind = self.bag.pop()
        shape = [(x, y) for y, row in enumerate(self.SHAPES[kind].split("/"))
                 for x, cell in enumerate(row) if cell == "#"]
        initial = shape[:]
        candidates = []
        for rotation in range(4):
            for x in range(self.WIDTH - max(dx for dx, dy in shape)):
                if not self._fits(shape, x, 0):
                    continue
                y = 0
                while self._fits(shape, x, y + 1):
                    y += 1
                trial = self.board.copy()
                for dx, dy in shape:
                    trial[y + dy, x + dx] = kind + 1
                full = np.all(trial != 0, axis=1)
                cleared = int(full.sum())
                trial = np.vstack((np.zeros((cleared, self.WIDTH), dtype=int), trial[~full]))
                occupied = trial != 0
                heights = [self.HEIGHT - int(np.argmax(col)) if col.any() else 0 for col in occupied.T]
                holes = sum(int((col[np.argmax(col):] == 0).sum()) for col in occupied.T if col.any())
                bump = sum(abs(a - b) for a, b in zip(heights, heights[1:]))
                score = cleared * 8 - sum(heights) * 0.5 - holes * 7 - bump * 0.35
                candidates.append((score, x, rotation))
            shape = self._rotate(shape)
        if not candidates or not self._fits(initial, 3, 0):
            self.piece = []
            self.pause = 24
            return
        self.rng.shuffle(candidates)
        _, self.target_x, self.turns = max(candidates, key=lambda item: item[0])
        self.piece = initial
        self.x, self.y, self.color = 3, 0, kind + 1

    def _step(self):
        # Enter at the centre, rotate, slide to the chosen column, then fall.
        # Every intermediate pose must fit; no teleporting into the stack.
        if self.turns:
            turned = self._rotate(self.piece)
            for kick in (0, -1, 1, -2, 2):
                if self._fits(turned, self.x + kick, self.y):
                    self.piece = turned
                    self.x += kick
                    self.turns -= 1
                    return
            self.turns = 0
            self.target_x = self.x
        if self.x != self.target_x:
            dx = 1 if self.target_x > self.x else -1
            if self._fits(self.piece, self.x + dx, self.y):
                self.x += dx
                return
            self.target_x = self.x
        if self._fits(self.piece, self.x, self.y + 1):
            self.y += 1
        else:
            self._lock()

    def _lock(self):
        for dx, dy in self.piece:
            self.board[self.y + dy, self.x + dx] = self.color
        self.piece = []
        self.clearing = list(np.flatnonzero(np.all(self.board != 0, axis=1)))
        if self.clearing:
            self.lines += len(self.clearing)
            self.pause = 8
        else:
            self._spawn()

    def next(self) -> list[RGB]:
        self.tick += 1
        if self.pause:
            self.pause -= 1
            if not self.pause:
                if self.clearing:
                    rows = np.delete(self.board, self.clearing, axis=0)
                    self.board = np.vstack((np.zeros((len(self.clearing), self.WIDTH), dtype=int), rows))
                    self.clearing = []
                else:
                    self.board.fill(0)
                self._spawn()
        elif self.tick % 2 == 0:
            self._step()
        frame = np.zeros((N, N, 3))
        frame[:15, 3:13] = np.array(self.COLORS)[self.board]
        frame[:, [2, 13]] = frame[15, 2:14] = (72, 72, 72)
        for dx, dy in self.piece:
            _dot(frame, self.x + dx + 3, self.y + dy, self.COLORS[self.color])
        for y in self.clearing:
            frame[y, 3:3 + min(10, (9 - self.pause) * 2)] = WHITE
        return _as_pixels(frame)


class Invaders(Effect):
    """Nine little marching aliens, steady waves, and a pilot that dodges fire."""

    name = "invaders"
    ALIEN = (".##.", "####", "#..#")
    WIDTH, HEIGHT = 4, 3
    COLORS = (GREEN, PINK, YELLOW)

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.tick = self.pause = 0
        self._reset()

    def _reset(self):
        self.aliens = [(x * 5, y * 4) for y in range(3) for x in range(3)]
        self.offset, self.drop, self.direction = 1, 0, 1
        self.tick = self.bounces = 0
        self.restart = False
        self.shield = 40
        self.ship = 7
        self.shots, self.bombs, self.blasts = [], [], []

    def next(self) -> list[RGB]:
        self.tick += 1
        self.shield = max(0, self.shield - 1)
        if self.pause:
            self.pause -= 1
            if not self.pause:
                if self.restart:
                    self._reset()
                else:
                    # A hit costs a ship, not the entire wave's progress.
                    self.ship, self.shield = 7, 40
                    self.shots, self.bombs = [], []
        else:
            if self.tick % 20 == 0:
                self.offset += self.direction
                if self.offset in (0, 2):
                    self.direction *= -1
                    self.bounces += 1
                    if self.bounces % 4 == 0:
                        self.drop += 1
            if self.aliens and self.tick % 4 == 0:
                target = min(self.aliens, key=lambda p: abs(p[0] + self.offset + self.WIDTH // 2 - self.ship))
                target_x = target[0] + self.offset + self.WIDTH // 2
                threats = [x for x, y in self.bombs if y >= 10]
                safe = [x for x in range(1, 15) if all(abs(x - bomb) > 1 for bomb in threats)]
                if safe and any(abs(self.ship - bomb) <= 1 for bomb in threats):
                    target_x = min(safe, key=lambda x: (abs(x - self.ship), abs(x - target_x)))
                self.ship += (target_x > self.ship) - (target_x < self.ship)
                self.ship = max(1, min(14, self.ship))
            if self.tick % 20 == 0:
                self.shots.append((self.ship, 13))
            if self.aliens and self.tick % 60 == 0:
                x, y = self.rng.choice(self.aliens)
                self.bombs.append((x + self.offset + self.WIDTH // 2, y + self.drop + self.HEIGHT))
            if self.tick % 2 == 0:
                live = []
                for x, y in self.shots:
                    y -= 1
                    hit = next((a for a in self.aliens
                                if a[0] + self.offset <= x < a[0] + self.offset + self.WIDTH
                                and a[1] + self.drop <= y < a[1] + self.drop + self.HEIGHT), None)
                    if hit is not None:
                        self.aliens.remove(hit)
                        self.blasts.append((x, y, 6))
                    elif y >= 0:
                        live.append((x, y))
                self.shots = live
            if self.tick % 4 == 0:
                self.bombs = [(x, y + 1) for x, y in self.bombs if y < 15]
            if not self.aliens or any(y + self.drop + self.HEIGHT - 1 >= 13 for x, y in self.aliens):
                self.restart = True
                self.pause = 20
            elif not self.shield and any(y >= 14 and abs(x - self.ship) <= 1 for x, y in self.bombs):
                self.restart = False
                self.pause = 12
                self.blasts.append((self.ship, 14, 6))
        frame = np.zeros((N, N, 3))
        for x, y in self.aliens:
            _sprite(frame, x + self.offset, y + self.drop, self.ALIEN,
                    self.COLORS[y // 4])
        for x, y in self.shots:
            _dot(frame, x, y, WHITE)
        for x, y in self.bombs:
            _dot(frame, x, y, RED)
        for x, y, ttl in self.blasts:
            for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                _dot(frame, x + dx, y + dy, YELLOW)
        self.blasts = [(x, y, ttl - 1) for x, y, ttl in self.blasts if ttl > 1]
        _sprite(frame, self.ship - 1, 14, (".#.", "###"), CYAN)
        return _as_pixels(frame)


class Pacman(Effect):
    """A native 16-pixel maze: dots, a chomping hero, and pursuing ghosts."""

    name = "pacman"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.cells = {(x, y) for y in range(2, 14) for x in range(2, 14)
                      if x in (2, 7, 13) or y in (2, 7, 13)}
        self.tick = self.pause = 0
        self._reset()

    def _reset(self):
        self.hero = (2, 2)
        self.ghosts = [(13, 13), (13, 2)]
        self.dots = {p for p in self.cells if sum(p) % 2 == 0} - {self.hero}
        self.direction = (1, 0)

    def next(self) -> list[RGB]:
        self.tick += 1
        if self.pause:
            self.pause -= 1
            if not self.pause:
                self._reset()
        else:
            if self.tick % 3 == 0:
                route = _path(self.hero, self.dots, set(self.ghosts), self.cells)
                if not route:
                    route = _path(self.hero, self.dots, set(), self.cells)
                if route:
                    nxt = route[0]
                    self.direction = (nxt[0] - self.hero[0], nxt[1] - self.hero[1])
                    self.hero = nxt
                    self.dots.discard(nxt)
                if self.hero in self.ghosts:
                    self.pause = 18
            if not self.pause and self.tick % 5 == 0:
                for index, ghost in enumerate(self.ghosts):
                    target = self.hero if self.rng.random() < 0.75 else self.rng.choice(sorted(self.cells))
                    route = _path(ghost, {target}, set(), self.cells)
                    if route:
                        self.ghosts[index] = route[0]
            if not self.dots or self.hero in self.ghosts:
                self.pause = 18
        frame = np.full((N, N, 3), (0, 0, 64), dtype=float)
        for x, y in self.cells:
            frame[y - 1:y + 2, x - 1:x + 2] = BLACK
        for x, y in self.dots:
            _dot(frame, x, y, (72, 72, 0))
        for (x, y), color in zip(self.ghosts, (RED, PINK)):
            _sprite(frame, x - 1, y - 1, (".#.", "###", "#.#"), color)
        x, y = self.hero
        _sprite(frame, x - 1, y - 1, (".#.", "###", ".#."), YELLOW)
        if (self.tick // 3) % 2 == 0:
            _dot(frame, x + self.direction[0], y + self.direction[1], BLACK)
        return _as_pixels(frame)


class PacFace(Effect):
    """A large chomping mascot, with passing pellets and an occasional blink."""

    name = "pacface"

    def __init__(self, seed: int = 0):
        self.tick = 0
        self.blink = 65 + random.Random(seed).randrange(50)
        ys, xs = np.mgrid[0:N, 0:N]
        self.dx, self.dy = xs - 6.5, ys - 7.5

    def next(self) -> list[RGB]:
        self.tick += 1
        opening = (0.5 - 0.5 * math.cos(self.tick * math.tau / 22)) * 0.95
        face = self.dx * self.dx + self.dy * self.dy <= 6 ** 2
        mouth = (self.dx > 0) & (abs(self.dy) < self.dx * opening)
        frame = np.zeros((N, N, 3))
        frame[face & ~mouth] = YELLOW
        # A two-pixel eye survives LED bloom; a one-pixel dimple does not.
        frame[4:6, 7:9] = BLACK
        if self.tick % 160 in range(self.blink, self.blink + 5):
            frame[4, 7:9] = YELLOW
        pellet = 15 - (self.tick % 22) * 0.35
        if opening > 0.2 and pellet > 9:
            _dot(frame, pellet, 7, WHITE)
            _dot(frame, pellet, 8, WHITE)
        return _as_pixels(frame)


class GhostFace(Effect):
    """A close-up arcade ghost with wandering pupils, blinks, and waving feet."""

    name = "ghostface"
    MASK = ("....######....", "..##########..", ".############.") + ("##############",) * 10

    def __init__(self, seed: int = 0):
        self.tick = 0
        self.phase = random.Random(seed).uniform(0, math.tau)

    def next(self) -> list[RGB]:
        self.tick += 1
        frame = np.zeros((N, N, 3))
        _sprite(frame, 1, 1, self.MASK, PINK)
        for x in range(1, 15):
            if (x + self.tick // 5) % 6 < 3:
                _dot(frame, x, 14, PINK)
        gaze = math.sin(self.tick * 0.045 + self.phase)
        dx = 0 if gaze < -0.3 else 2 if gaze > 0.3 else 1
        for x in (3, 9):
            frame[5:9, x:x + 4] = WHITE
            frame[6:8, x + dx:x + dx + 2] = BLACK
            if self.tick % 115 in (105, 106, 107):
                frame[5:7, x:x + 4] = PINK
                frame[8, x:x + 4] = PINK
        return _as_pixels(frame)


class Tron(Effect):
    """Four fast light cycles, unpredictable turns, fading trails, and crashes."""

    name = "tron"
    COLORS = (CYAN, PINK, GREEN, YELLOW)
    TRAIL_LIFE = 32

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.tick = 0
        self._reset()

    def _reset(self):
        self.heads = [(1, 5), (14, 10), (5, 14), (10, 1)]
        self.directions = [(1, 0), (-1, 0), (0, -1), (0, 1)]
        self.trails = {p: i for i, p in enumerate(self.heads)}
        self.ages = {p: self.TRAIL_LIFE for p in self.heads}
        self.waits = [0] * 4
        self.blasts = []

    def _respawn(self, index):
        free = [(x, y) for y in range(N) for x in range(N) if (x, y) not in self.trails]
        if not free:
            return
        self.heads[index] = self.rng.choice(free)
        self.directions[index] = self.rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        self.trails[self.heads[index]] = index
        self.ages[self.heads[index]] = self.TRAIL_LIFE

    def _crash(self, index):
        self.blasts.append((*self.heads[index], 10, index))
        self.waits[index] = 6

    def next(self) -> list[RGB]:
        self.tick += 1
        if self.tick % 2 == 0:
            self.ages = {p: age - 1 for p, age in self.ages.items() if age > 1}
            self.trails = {p: owner for p, owner in self.trails.items() if p in self.ages}
            moves = {}
            for index, (x, y) in enumerate(self.heads):
                if self.waits[index]:
                    self.waits[index] -= 1
                    if not self.waits[index]:
                        self._respawn(index)
                    continue
                dx, dy = self.directions[index]
                choices = [(dx, dy), (-dy, dx), (dy, -dx)]
                safe = [(a, b) for a, b in choices if 0 <= x + a < N and 0 <= y + b < N
                        and (x + a, y + b) not in self.trails]
                if not safe:
                    self._crash(index)
                    continue
                direction = safe[0] if self.rng.random() > 0.55 else self.rng.choice(safe)
                self.directions[index] = direction
                moves[index] = (x + direction[0], y + direction[1])
            for index, point in moves.items():
                if list(moves.values()).count(point) > 1 or point in self.trails:
                    self._crash(index)
                else:
                    self.heads[index] = point
                    self.trails[point] = index
                    self.ages[point] = self.TRAIL_LIFE
        frame = np.zeros((N, N, 3))
        for (x, y), owner in self.trails.items():
            color = np.array(self.COLORS[owner]) * (0.2 + 0.65 * self.ages[x, y] / self.TRAIL_LIFE)
            _dot(frame, x, y, color)
        for index, (x, y) in enumerate(self.heads):
            if not self.waits[index]:
                _dot(frame, x, y, WHITE)
        for x, y, ttl, owner in self.blasts:
            spread = 1 if ttl > 5 else 2
            for dx, dy in ((-spread, 0), (spread, 0), (0, -spread), (0, spread)):
                _dot(frame, x + dx, y + dy, np.array(self.COLORS[owner]) * ttl / 10)
        self.blasts = [(x, y, ttl - 1, owner) for x, y, ttl, owner in self.blasts if ttl > 1]
        return _as_pixels(frame)


class Hyperspace(Effect):
    """Cruise, accelerate into blue-white streaks, coast, and drop out of warp."""

    name = "hyperspace"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.tick = 0
        self.stars = [self._spawn(self.rng.uniform(0.1, 1)) for _ in range(32)]
        self.ys, self.xs = np.mgrid[0:N, 0:N]

    def _spawn(self, depth=1.0):
        angle = self.rng.uniform(0, math.tau)
        radius = self.rng.uniform(0.25, 1.4)
        return [math.cos(angle) * radius, math.sin(angle) * radius, depth]

    def next(self) -> list[RGB]:
        phase = self.tick % 240
        warp = max(0.0, min(1.0, (phase - 50) / 45, (220 - phase) / 45))
        warp = warp * warp * (3 - 2 * warp)
        self.tick += 1
        speed = 0.008 + warp * 0.07
        frame = np.zeros((N, N, 3))
        for star in self.stars:
            star[2] -= speed
            if star[2] <= 0.06:
                star[:] = self._spawn()
            sx, sy, z = star
            x, y = 7.5 + sx * 4 / z, 7.5 + sy * 4 / z
            tail_z = z + speed * (1 + warp * 2)
            tx, ty = 7.5 + sx * 4 / tail_z, 7.5 + sy * 4 / tail_z
            dx, dy = x - tx, y - ty
            along = np.clip(((self.xs - tx) * dx + (self.ys - ty) * dy) /
                            max(dx * dx + dy * dy, 1e-9), 0, 1)
            distance = np.hypot(self.xs - tx - along * dx, self.ys - ty - along * dy)
            coverage = np.clip(0.85 - distance, 0, 1)
            light = coverage * min(1.0, 0.25 + (1 - z)) * (0.4 + along * 0.6)
            tone = np.array((255 - warp * 140, 255 - warp * 60, 255))
            frame = np.maximum(frame, light[..., None] * tone)
        return _as_pixels(frame)


class Tunnel(Effect):
    """A spiralling cylindrical tunnel with flowing bands and a dark centre."""

    name = "tunnel"

    def __init__(self, seed: int = 0):
        self.t = seed * 0.7
        self.ys, self.xs = np.mgrid[0:N, 0:N]

    def next(self) -> list[RGB]:
        self.t += 0.035
        t = self.t
        x = self.xs - 7.5 - math.sin(t * 0.7) * 1.5
        y = self.ys - 7.5 - math.cos(t * 0.5) * 1.5
        radius = np.hypot(x, y)
        angle = np.arctan2(y, x)
        depth = 9 / (radius + 1.5) + t * 0.8
        twist = angle * 3 + depth * 3 - t * 0.5
        band = (np.sin(twist) + 1) / 2
        light = (0.08 + band ** 5 * 0.92) * np.clip((radius - 1.2) / 4, 0, 1)
        frame = _hsv(0.52 + (1 - band) * 0.24, 1.0, light)
        return _as_pixels(frame)


class Streams(Effect):
    """A panning network that grows links and carries packets between glowing nodes.

    Geometry lives on a world larger than the panel. The camera wraps over a
    periodic graph, so the view can drift indefinitely without an edge or a cut.
    Lines and node halos use analytic coverage at each LED, not resized artwork.
    """

    name = "streams"
    STYLE = "grid"
    SPACING = 14
    PALETTE = (CYAN, PINK)

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.t = 0.0
        self.world = self.SPACING * 8
        self.origin = (self.rng.uniform(-1, 1), self.rng.uniform(-1, 1))
        self.ys, self.xs = np.mgrid[0:N, 0:N]
        nodes = {}
        for y in range(8):
            for x in range(8):
                jitter = 1.1 if self.STYLE == "web" else 0
                nodes[x, y] = np.array((x * self.SPACING + self.rng.uniform(-jitter, jitter),
                                       y * self.SPACING + self.rng.uniform(-jitter, jitter)))
        self.edges = []
        for (gx, gy), start in nodes.items():
            directions = ((1, 0), (0, 1))
            if self.STYLE == "web":
                directions = ((1, 0), (0, 1), (1, 1))
            elif self.STYLE == "lanes":
                directions = ((1, 0), (1, 1 if gx % 2 else -1))
            for dx, dy in directions:
                if (dx, dy) != (1, 0) and self.rng.random() < 0.16:
                    continue
                end = nodes[(gx + dx) % 8, (gy + dy) % 8]
                delta = (end - start + self.world / 2) % self.world - self.world / 2
                points = [np.zeros(2)]
                if self.STYLE == "circuit":
                    # A dogleg distinguishes circuit traces from the square mesh.
                    if dx:
                        points += [np.array((delta[0] / 2, -1.5)), np.array((delta[0] / 2 + 1.5, 0))]
                    else:
                        points += [np.array((1.5, delta[1] / 2)), np.array((0, delta[1] / 2 + 1.5))]
                elif self.STYLE == "lanes" and dy:
                    points += [np.array((delta[0] / 2, 0)), np.array((delta[0] / 2, delta[1]))]
                points.append(delta)
                color = np.array(self.PALETTE[self.rng.randrange(len(self.PALETTE))])
                self.edges.append((start, points, color, self.rng.uniform(0, 1.8)))

    def _line(self, a, b):
        delta = b - a
        along = np.clip(((self.xs - a[0]) * delta[0] + (self.ys - a[1]) * delta[1]) /
                        max(float(delta @ delta), 1e-9), 0, 1)
        distance = np.hypot(self.xs - a[0] - along * delta[0], self.ys - a[1] - along * delta[1])
        return np.clip(1.2 - distance, 0, 1)

    def _glow(self, point, strength):
        distance = np.hypot(self.xs - point[0], self.ys - point[1])
        core = np.clip(1.3 - distance, 0, 1)
        halo = np.exp(-distance * distance / 3.5) * strength
        return core, halo

    def next(self) -> list[RGB]:
        self.t += 0.05
        t = self.t
        camera = np.array(self.origin) + np.array((t * 0.65, t * 0.4))
        if self.STYLE == "lanes":
            camera = np.array(self.origin) + np.array((t * 0.85, math.sin(t * 0.2) * 2))
        frame = np.zeros((N, N, 3))
        for start, points, color, phase in self.edges:
            anchor = (start - camera + self.world / 2) % self.world - self.world / 2 + 7.5
            margin = self.SPACING + 3
            if not -margin < anchor[0] < N + margin or not -margin < anchor[1] < N + margin:
                continue
            clock = (t + phase) % 10
            growth = min(1, (clock + 0.4) / 2.6)
            fade = min(1, (10 - clock) / 0.7)
            lengths = [float(np.linalg.norm(b - a)) for a, b in zip(points, points[1:])]
            total = sum(lengths)
            built = growth * total
            packet = min(growth, (clock * 0.42) % 1) * total
            covered = 0.0
            for a, b, length in zip(points, points[1:], lengths):
                fraction = max(0, min(1, (built - covered) / length))
                if fraction > 0:
                    ink = self._line(anchor + a, anchor + a + (b - a) * fraction)
                    frame = np.maximum(frame, ink[..., None] * color * 0.85 * fade)
                if covered <= packet <= covered + length:
                    position = anchor + a + (b - a) * ((packet - covered) / length)
                    core, halo = self._glow(position, 0.75)
                    frame = np.maximum(frame, halo[..., None] * color * fade)
                    frame = np.maximum(frame, core[..., None] * np.array(WHITE) * fade)
                covered += length
            # Arrival expands into a small halo at the receiving intersection.
            arrival = max(0, (packet / total - 0.72) / 0.28)
            for point, strength in ((anchor, 0.3), (anchor + points[-1], 0.3 + arrival * 0.7)):
                core, halo = self._glow(point, strength)
                frame = np.maximum(frame, halo[..., None] * color * fade)
                frame = np.maximum(frame, core[..., None] * color * 0.95 * fade)
        return _as_pixels(frame)


class Circuit(Streams):
    """Angular green/gold circuit traces with travelling signals."""

    name = "circuit"
    STYLE = "circuit"
    SPACING = 13
    PALETTE = (GREEN, YELLOW)


class Metro(Streams):
    """Parallel data lanes, switching links, and a lateral camera drift."""

    name = "metro"
    STYLE = "lanes"
    SPACING = 11
    PALETTE = (CYAN, YELLOW)


class Synapses(Streams):
    """A loose triangular web of violet/cyan nodes and firing connections."""

    name = "synapses"
    STYLE = "web"
    SPACING = 13
    PALETTE = (PINK, CYAN)


ARCADE = {e.name: e for e in (Snake, Pong, Breakout, Tetris, Invaders, PacFace, GhostFace, Tron,
                            Hyperspace, Tunnel)}
NETWORKS = {e.name: e for e in (Streams, Circuit, Metro, Synapses)}
ABSTRACT = ({e.name: e for e in (Plasma, Fire, Life, Rain, Starfield, Bounce, Rings, Pacman)}
            | ARCADE | NETWORKS)


def build(name: str, seed: int = 0) -> Effect:
    if name not in ABSTRACT:
        raise ValueError(f"unknown effect {name!r}; try {', '.join(sorted(ABSTRACT))}")
    return ABSTRACT[name](seed)  # type: ignore[call-arg]
