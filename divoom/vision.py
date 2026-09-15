"""Read the LED panel back through a webcam, so the test loop can verify itself.

Calibration makes the panel spell out red, green, then blue: only the matrix can
do that in one place, which survives a person moving around in frame. No hardcoded
coordinates, so moving the speaker only costs a recalibration.

Recovering a colour takes three corrections, each of which was wrong at first:
the panel is glossy and mirrors the room, the camera mixes the channels, and
neither composes correctly outside linear light.
"""

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .proto import RGB, SCREEN

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "bin" / "snapshot"
CAPTURES = ROOT / "captures"
PANEL_FILE = CAPTURES / "panel.json"

GAMMA = 2.2
IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


class VisionError(RuntimeError):
    pass


def _linear(color) -> np.ndarray:
    return (np.asarray(color, dtype=float) / 255.0) ** GAMMA


def _encode(linear: np.ndarray) -> np.ndarray:
    return np.clip(linear, 0, 1) ** (1 / GAMMA) * 255


@dataclass
class Panel:
    """Where every LED sits in the frame, and how to undo the camera.

    Two things make a per-axis grid wrong. The panel is never perfectly square
    to the camera, so rows drift as you cross the screen — hence interpolating
    inside the corner quad. And the lit area spans 15 pitches plus one LED, not
    16 pitches, so splitting it into 16 equal cells loses a row by the far edge.
    """

    points: list[list[int]]
    radius: int
    matrix: list[list[float]] = field(default_factory=lambda: [row[:] for row in IDENTITY])
    black: list[list[int]] = field(default_factory=list)

    @property
    def box(self) -> tuple[int, int, int, int]:
        xs = [x for x, _ in self.points]
        ys = [y for _, y in self.points]
        return (min(xs), min(ys), max(xs), max(ys))

    def save(self) -> None:
        PANEL_FILE.write_text(json.dumps(self.__dict__))

    @classmethod
    def load(cls) -> "Panel":
        if not PANEL_FILE.exists():
            raise VisionError("panel not calibrated yet — run `calibrate`")
        return cls(**json.loads(PANEL_FILE.read_text()))

    def correct(self, raw: list[RGB]) -> list[RGB]:
        """Room reflection out, camera crosstalk out, both in linear light.

        Subtracting the dark reference straight off the sRGB values looked
        equivalent and is not: it turns a white screen blue, because a neutral
        grey reflection is only an additive offset once gamma is undone.
        """
        lit = _linear(raw)
        if self.black:
            lit = np.maximum(lit - _linear(self.black), 0.0)
        out = np.maximum(lit @ np.asarray(self.matrix).T, 0.0)

        peak = out.max(axis=1, keepdims=True)
        out = np.where(peak > 1, out / np.maximum(peak, 1e-9), out)
        return [tuple(int(v) for v in px) for px in _encode(out)]  # type: ignore[misc]


def snapshot(name: str, device: int = 0, warmup: int = 12) -> Image.Image:
    CAPTURES.mkdir(exist_ok=True)
    path = CAPTURES / name
    done = subprocess.run(
        [str(SNAPSHOT), str(path), "--device", str(device), "--warmup", str(warmup)],
        capture_output=True, text=True, timeout=60,
    )
    if done.returncode != 0:
        raise VisionError(done.stderr.strip() or "snapshot failed")
    return Image.open(path).convert("RGB")


def sample(image: Image.Image, panel: Panel) -> list[RGB]:
    """Average a small patch of raw pixels around each LED centre."""
    frame = np.asarray(image, dtype=np.int32)
    r = panel.radius
    out: list[RGB] = []
    for cx, cy in panel.points:
        patch = frame[max(cy - r, 0):cy + r + 1, max(cx - r, 0):cx + r + 1]
        mean = patch.reshape(-1, 3).mean(axis=0)
        out.append((int(mean[0]), int(mean[1]), int(mean[2])))
    return out


def read(image: Image.Image, panel: Panel) -> list[RGB]:
    return panel.correct(sample(image, panel))


def _dominant(image: Image.Image, channel: int, floor: int = 70, lead: float = 1.35):
    """Pixels where one channel clearly outruns the other two."""
    a = np.asarray(image, dtype=np.int16)
    mine = a[:, :, channel]
    others = np.delete(a, channel, axis=2).max(axis=2)
    return (mine > floor) & (mine > others * lead)


def _densest_run(profile: np.ndarray, share: float = 0.2) -> tuple[int, int]:
    """Longest stretch where the mask is dense, which drops faint reflections."""
    hot = profile > profile.max() * share
    best = run = None
    for i, on in enumerate(hot):
        if on:
            run = (i, i) if run is None else (run[0], i)
            if best is None or (run[1] - run[0]) > (best[1] - best[0]):
                best = run
        else:
            run = None
    if best is None:
        raise VisionError("no dense region found")
    return best


def _corners(ys: np.ndarray, xs: np.ndarray) -> list[tuple[float, float]]:
    """Quad corners of a point cloud, via the extremes of x+y and x-y."""
    total, diff = xs + ys, xs - ys
    pick = (total.argmin(), diff.argmax(), total.argmax(), diff.argmin())
    return [(float(xs[i]), float(ys[i])) for i in pick]


def locate(red: Image.Image, green: Image.Image, blue: Image.Image,
           white: Image.Image, dark: Image.Image) -> Panel:
    """Find the matrix by making it spell out a colour sequence.

    Differencing against a black frame seemed obvious but fails in a room with a
    person in it: they move between shots and their silhouette dominates the diff.
    Only the panel can be red, then green, then blue in the same place.
    """
    mask = _dominant(red, 0) & _dominant(green, 1) & _dominant(blue, 2)
    if mask.sum() < 200:
        raise VisionError(f"only {mask.sum()} px followed red→green→blue — "
                          "is the panel in shot and bright enough?")

    # Keep only the densest band on each axis, so a reflection on a nearby
    # surface cannot stretch the quad.
    left, right = _densest_run(mask.sum(axis=0))
    top, bottom = _densest_run(mask.sum(axis=1))
    trimmed = np.zeros_like(mask)
    trimmed[top:bottom + 1, left:right + 1] = mask[top:bottom + 1, left:right + 1]

    width, height = right - left, bottom - top
    if min(width, height) < 40:
        raise VisionError(f"lit area too small ({width}x{height}px) to read reliably")
    if not 0.6 < width / height < 1.7:
        raise VisionError(f"panel reads as {width}x{height}px, which is not square enough")

    ys, xs = np.nonzero(trimmed)
    tl, tr, br, bl = _corners(ys, xs)

    # An LED covers `duty` of its pitch; the quad spans 15 pitches plus one LED,
    # so the first centre sits half an LED in rather than half a cell in.
    duty = float(np.clip(np.sqrt(trimmed.sum() / (width * height)), 0.35, 0.95))
    steps = [(duty / 2 + i) / (SCREEN - 1 + duty) for i in range(SCREEN)]

    points = []
    for v in steps:
        for u in steps:
            x = ((1 - u) * (1 - v) * tl[0] + u * (1 - v) * tr[0]
                 + u * v * br[0] + (1 - u) * v * bl[0])
            y = ((1 - u) * (1 - v) * tl[1] + u * (1 - v) * tr[1]
                 + u * v * br[1] + (1 - u) * v * bl[1])
            points.append([round(x), round(y)])

    panel = Panel(points, max(1, round(min(width, height) / (SCREEN - 1) * duty * 0.4)))
    panel.black = [list(c) for c in sample(dark, panel)]
    panel.matrix = _colour_solve(panel, red, green, blue, white)
    return panel


def _colour_solve(panel: Panel, red: Image.Image, green: Image.Image,
                  blue: Image.Image, white: Image.Image) -> list[list[float]]:
    """Solve the 3x3 that maps what the camera sees back to what we sent.

    A green LED photographs with a fifth of its energy in the red and blue
    channels — sensor crosstalk, chroma subsampling and bloom together. Widening
    the comparison threshold instead would have hidden real errors.

    The primaries alone are not enough. The panel's white is not the sum of its
    three primaries at full power — it limits current with all three subpixels
    lit — and the webcam re-exposes between a red frame and a white one. So
    normalise the result against the panel's own white, which is what the eye
    does anyway: colours are judged relative to the white in front of it.
    """
    black = _linear(panel.black)

    def net(shot: Image.Image) -> np.ndarray:
        values = np.maximum(_linear(sample(shot, panel)) - black, 0.0)
        return values[values.max(axis=1) > 0.02].mean(axis=0)

    observed = np.array([net(red), net(green), net(blue)]).T
    try:
        matrix = np.linalg.inv(observed)
    except np.linalg.LinAlgError:
        return [row[:] for row in IDENTITY]

    balance = matrix @ net(white)
    if np.all(balance > 1e-6):
        matrix = np.diag(1 / balance) @ matrix
    return matrix.tolist()


SYMBOLS = {
    (True, False, False): "R",
    (False, True, False): "G",
    (False, False, True): "B",
    (True, True, False): "Y",
    (False, True, True): "C",
    (True, False, True): "M",
    (True, True, True): "W",
}


def _classify(color: RGB, lit_floor: int) -> str:
    peak = max(color)
    if peak < lit_floor:
        return "."
    # A camera never returns clean primaries, so a channel counts as "on" when it
    # is within a fraction of the strongest one rather than at some fixed level.
    lead = tuple(c > peak * 0.72 for c in color)
    return SYMBOLS[lead]  # type: ignore[index]


def classify_grid(colors: list[RGB], lit_floor: int = 55) -> list[str]:
    return [_classify(c, lit_floor) for c in colors]


def expected_grid(pixels: list[RGB], lit_floor: int = 40) -> list[str]:
    return [_classify(c, lit_floor) for c in pixels]


def render(symbols: list[str]) -> str:
    tint = {"R": "31", "G": "32", "B": "34", "Y": "33", "C": "36", "M": "35", "W": "37", ".": "90"}
    rows = []
    for y in range(SCREEN):
        row = symbols[y * SCREEN:(y + 1) * SCREEN]
        rows.append("".join(f"\x1b[{tint[s]}m{s}\x1b[0m" for s in row))
    return "\n".join(rows)


def agreement(expected: list[str], observed: list[str]) -> float:
    hits = sum(1 for a, b in zip(expected, observed) if a == b)
    return hits / len(expected)


def similarity(expected: list[RGB], observed: list[RGB], lit_floor: int = 55) -> float:
    """Per-LED score on channel ratios rather than on a seven-symbol bucket.

    Bucketing reports a correct smooth gradient as a failure, because adjacent
    LEDs differ by far less than the gap between two symbols. Hue alone is no
    better: it is undefined for white and grey, so a correct white screen scores
    near zero. Ratios handle saturated and neutral colours alike, and ignore the
    overall brightness the camera never reproduces. Structural errors still
    score zero — a LED that should be dark and is not has no colour to compare.
    """
    total = 0.0
    for want, got in zip(expected, observed):
        want_lit, got_lit = max(want) >= 40, max(got) >= lit_floor
        if not want_lit and not got_lit:
            total += 1.0
        elif want_lit != got_lit:
            continue
        else:
            a = [v / max(want) for v in want]
            b = [v / max(got) for v in got]
            drift = sum(abs(x - y) for x, y in zip(a, b)) / 3
            total += max(0.0, 1 - drift / 0.5)
    return total / len(expected)
