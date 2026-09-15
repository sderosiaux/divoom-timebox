"""Status-bar faces, in the spirit of Doom's mugshot.

Two kinds of character, because two kinds of source material:

`SpriteFace` downscales real 24x29 sprite sheets. Drawing the marine by hand
from memory produced something only vaguely Doom-shaped — the proportions of a
face are exactly what memory gets wrong. Working from the artwork fixes that,
and gets the gaze frames, the wound progression and the death face for free.

`AsciiFace` keeps hand-authored grids for characters with no sprite sheet. At
16x16 an ASCII grid stays readable and editable in a diff, which beats a binary
blob when the only reference is a photograph.
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image

from .proto import RGB, SCREEN

ASSETS = Path(__file__).resolve().parent.parent / "assets"

# Health at or above which each sprite tier applies, healthiest first.
TIERS = (80, 60, 40, 20, 1)


def tier_of(health: int) -> int:
    for index, floor in enumerate(TIERS):
        if health >= floor:
            return index
    return len(TIERS) - 1


class Character:
    name = "character"

    def frame(self, health: int = 100, gaze: int = 0, blink: bool = False,
              mood: str = "idle") -> list[RGB]:
        raise NotImplementedError


class SpriteFace(Character):
    """A Doom-style sprite sheet, squashed into the panel.

    Preserving the 24:29 aspect and cropping loses the mouth, which carries most
    of the expression; squashing the whole head keeps hair, eyes and mouth, and
    at this size nobody reads the distortion as distortion.
    """

    # Sprite variant per gaze: pupils sit left in `2`, centred in `1`, right in `0`.
    GAZE = {-1: "2", 0: "1", 1: "0"}

    # Five tones, spread so that adjacent ones survive the panel's bloom.
    RAMP = ((22, 9, 4), (98, 42, 14), (192, 94, 30), (246, 172, 98), (255, 238, 204))
    BLOOD = (236, 30, 30)
    GLOW = (255, 216, 40)

    def __init__(self, name: str, folder: str, dark_bias: float = 0.8,
                 ramp=RAMP, blood_tone=BLOOD, glow_tone=GLOW):
        self.name = name
        self.folder = ASSETS / folder
        self.dark_bias = dark_bias
        self.ramp = ramp
        self.blood_tone = blood_tone
        self.glow_tone = glow_tone
        self._cache: dict[str, list[RGB]] = {}

    def _shrink(self, image: Image.Image) -> Image.Image:
        """Downscale while defending the dark features.

        An eye is one or two pixels wide in the source. Averaging a cell — what
        any ordinary resize does — dissolves it into the surrounding skin, and
        the face arrives as an orange smudge. So when a cell hides something far
        darker than its average, that darker pixel wins.
        """
        source = np.asarray(image, dtype=float)
        height, width, _ = source.shape
        rows = np.linspace(0, height, SCREEN + 1).round().astype(int)
        cols = np.linspace(0, width, SCREEN + 1).round().astype(int)

        out = np.zeros((SCREEN, SCREEN, 3))
        for j in range(SCREEN):
            for i in range(SCREEN):
                cell = source[rows[j]:max(rows[j + 1], rows[j] + 1),
                              cols[i]:max(cols[i + 1], cols[i] + 1)].reshape(-1, 3)
                mean = cell.mean(axis=0)
                darkest = cell[cell.sum(axis=1).argmin()]
                if mean.sum() - darkest.sum() > 150:
                    out[j, i] = darkest * self.dark_bias + mean * (1 - self.dark_bias)
                else:
                    out[j, i] = mean
        return Image.fromarray(out.clip(0, 255).astype("uint8"))

    def _load(self, sprite: str) -> list[RGB]:
        if sprite not in self._cache:
            path = self.folder / f"{sprite}.png"
            if not path.exists():
                path = self.folder / "stfst01.png"

            image = self._posterize(self._shrink(Image.open(path).convert("RGB")))
            self._cache[sprite] = list(image.getdata())  # type: ignore[arg-type]
        return self._cache[sprite]

    def _posterize(self, image: Image.Image) -> Image.Image:
        """Rebuild the face on a deliberately spread ramp.

        The sprite lives in a narrow band of browns and oranges. Faithful on a
        monitor, mush on an LED panel that blooms: neighbouring tones stop being
        distinguishable and the face reads as a blob. So throw the original hues
        away and map luminance onto five tones chosen to be far apart. Blood and
        the invulnerability glow carry meaning rather than shading, so they keep
        their own colour instead of joining the ramp.
        """
        source = np.asarray(image, dtype=float)
        red, green, blue = source[..., 0], source[..., 1], source[..., 2]
        luminance = source.sum(axis=2)
        face = luminance > 40
        if face.sum() < 10:
            return image

        low, high = np.percentile(luminance[face], (2, 98))
        steps = len(self.ramp)
        level = np.clip(((luminance - low) / max(high - low, 1) * steps).astype(int), 0, steps - 1)

        out = np.zeros_like(source)
        for index, tone in enumerate(self.ramp):
            out[(level == index) & face] = tone
        out[(red > green * 1.8) & (red > 90) & face] = self.blood_tone
        out[(red > blue * 1.8) & (green > blue * 1.6) & (green > 120) & face] = self.glow_tone
        return Image.fromarray(out.clip(0, 255).astype("uint8"))

    def sprite_for(self, health: int, gaze: int, mood: str) -> str:
        if health <= 0:
            return "stfdead0"
        if mood == "god":
            return "stfgod0"
        tier = tier_of(health)
        if mood in ("ouch", "evil", "kill"):
            return f"stf{'evl' if mood == 'evil' else mood}{tier}"
        return f"stfst{tier}{self.GAZE[gaze]}"

    def frame(self, health: int = 100, gaze: int = 0, blink: bool = False,
              mood: str = "idle") -> list[RGB]:
        # Doom's marine never blinks; there is no sprite for it and faking one
        # would look worse than the original.
        return self._load(self.sprite_for(health, gaze, mood))


class AsciiFace(Character):
    """Hand-drawn grid, with the pupils painted in per gaze."""

    base: list[str] = []
    eyes: tuple[tuple[int, int], tuple[int, int]] = ((0, 0), (0, 0))
    eye_rows: tuple[int, int] = (0, 0)
    mouth_row: int = 0
    mouths: dict[str, str] = {}
    blood: list[tuple[int, int]] = []
    palette: dict[str, RGB] = {}
    shut: str = "S"          # what an eye becomes when closed
    pale: dict[str, str] = {"s": "S", "S": "x", "n": "x"}

    def frame(self, health: int = 100, gaze: int = 0, blink: bool = False,
              mood: str = "idle") -> list[RGB]:
        grid = [list(row) for row in self.base]

        if health <= 0:
            mouth = "dead"
        elif mood == "ouch":
            mouth = "pain"
        elif mood in ("evil", "god"):
            mouth = "grin"
        else:
            mouth = ("set", "calm", "calm", "pain", "pain")[tier_of(health)]
        grid[self.mouth_row] = list(self.mouths[mouth])

        dead = health <= 0
        if dead:
            grid = [[self.pale.get(c, c) for c in row] for row in grid]

        spots = (0, 3, 6, 12, 16)[tier_of(health)] if not dead else len(self.blood)
        for x, y in self.blood[:spots]:
            grid[y][x] = "B" if (x + y) % 3 == 0 else "b"

        top, bottom = self.eye_rows
        for left, right in self.eyes:
            if blink or dead:
                for row in (top, bottom):
                    for x in range(left, right + 1):
                        grid[row][x] = "x" if dead else self.shut
                continue
            column = {-1: left, 0: (left + right) // 2, 1: right}[gaze]
            for row in (top, bottom):
                grid[row][column] = "p"

        return [self.palette.get(c, (0, 0, 0)) for row in grid for c in row]


DOOM_PALETTE: dict[str, RGB] = {
    ".": (0, 0, 0),
    # Measured on the panel rather than chosen by eye: against this skin, a
    # mid-brown separates by 188 while a near-black brown separates by 297. The
    # worry that dark hair would simply switch the LEDs off is unfounded — it
    # still reads as lit, just far enough down to give the head a top.
    "h": (38, 18, 5),        # hair, outer
    "H": (70, 32, 8),        # hair
    "s": (226, 152, 104),    # skin — one flat tone, no shading noise
    "S": (158, 92, 54),      # skin, only where a shape needs defining
    "w": (176, 162, 138),    # sclera, warm and dim: neutral grey photographs blue
    "p": (8, 8, 12),         # pupil
    "m": (74, 22, 22),       # mouth
    "t": (230, 230, 230),
    "a": (44, 78, 44),       # armour
    "A": (66, 108, 66),
    "b": (160, 14, 14),
    "B": (226, 30, 30),
    "x": (74, 40, 34),
}


class DoomGuy(AsciiFace):
    """The marine, redrawn for 16x16.

    Downscaling the real sprite was the obvious move and it fails: a 24x29 face
    carries shading the panel cannot resolve, so it arrives as scattered orange
    and white with no silhouette. Pixel artists redraw at the target size rather
    than resample, so this keeps the sprite's proportions — hair mass, eye line,
    mouth height — and throws away every tone that only existed to shade.
    """

    name = "doom"
    palette = DOOM_PALETTE

    base = [
        "....hhhhhhhh....",
        "..hhHHHHHHHHhh..",
        ".hHHHHHHHHHHHHh.",
        ".hHHHHHHHHHHHHh.",
        ".hHssssssssssHh.",
        ".hssssssssssssh.",
        ".sswwwsssswwwss.",
        ".sswwwsssswwwss.",
        ".ssssssSSssssss.",
        ".sssssSSSSsssss.",
        ".ssssssssssssss.",
        ".ssSmmmmmmmmSss.",
        "..ssssssssssss..",
        "...aaaaaaaaaa...",
        ".AaaaaaaaaaaaaA.",
        ".Aa..aaaaaa..aA.",
    ]

    eye_rows = (6, 7)
    eyes = ((3, 5), (10, 12))
    mouth_row = 11

    mouths = {
        "set":   ".ssSmmmmmmmmSss.",
        "calm":  ".sssmmmmmmmmsss.",
        "grin":  ".ssmttttttttmss.",
        "pain":  ".smmmmmmmmmmmms.",
        "dead":  ".smmmmmmmmmmmms.",
    }

    blood = [
        (6, 4), (5, 4), (6, 3),
        (10, 4), (9, 4), (10, 3),
        (2, 8), (2, 9), (3, 9),
        (13, 8), (13, 9), (12, 9),
        (4, 10), (11, 10), (5, 12), (10, 12),
        (7, 2), (8, 2), (1, 6), (14, 7),
    ]


CARMA_PALETTE: dict[str, RGB] = {
    ".": (0, 0, 0),
    "h": (104, 18, 8),
    "H": (208, 54, 18),
    "s": (242, 198, 170),
    "S": (148, 116, 98),
    "g": (30, 30, 36),
    "w": (12, 12, 18),
    "p": (255, 255, 255),    # glint on the glass
    "n": (198, 152, 126),
    "m": (96, 30, 30),
    "t": (244, 244, 244),
    "L": (72, 56, 44),
    "l": (44, 34, 28),
    "b": (150, 16, 16),
    "B": (216, 34, 34),
    "x": (74, 54, 48),
}


class MaxDamage(AsciiFace):
    """Carmageddon's driver. No visible eyes, so the gaze rides the lens glint."""

    name = "max"
    palette = CARMA_PALETTE
    shut = "w"

    base = [
        "..H..HH..HH..H..",
        ".HHHHHHHHHHHHHH.",
        ".HhhHHHHHHHHhhH.",
        ".HhssssssssssHh.",
        ".hsssssssssssss.",
        ".sggggggggggggs.",
        ".HgwwwggggwwwgH.",
        ".HgwwwggggwwwgH.",
        ".sgggggnnggggss.",
        "..ssssssnnssss..",
        "..ssSSSSSSSSss..",
        "..sSSsmmmmSSSs..",
        "...SSSSSSSSSS...",
        "..LLLLLLLLLLLL..",
        ".LllllllllllllL.",
        ".Ll..llllll..lL.",
    ]

    eye_rows = (6, 7)
    eyes = ((3, 5), (10, 12))
    mouth_row = 11

    mouths = {
        "set":   "..sSSsmmmmSSSs..",
        "calm":  "..sSmmmmmmmmSs..",
        "grin":  "..sSmttttttmSs..",
        "pain":  "..smmmmmmmmmms..",
        "dead":  "..smmmmmmmmmms..",
    }

    blood = [
        (6, 4), (5, 4), (6, 3),
        (10, 4), (11, 4), (10, 3),
        (2, 9), (3, 9), (2, 10),
        (13, 9), (12, 9), (13, 10),
        (4, 11), (11, 11), (5, 12), (10, 12),
        (7, 2), (8, 2), (1, 5), (14, 5),
    ]


SHADES_PALETTE: dict[str, RGB] = {
    ".": (0, 0, 0),
    # Measured against her skin on the panel: the light blonde first chosen
    # separates by 101, this warm venetian tone by 276 — and it happens to match
    # the photo better than the lighter one did.
    "h": (110, 52, 18),      # hair, in shadow
    "H": (182, 96, 36),      # strawberry blonde
    "s": (248, 210, 184),
    "S": (198, 158, 132),
    "g": (26, 26, 30),       # frame
    "w": (14, 14, 20),       # lens
    "p": (232, 232, 242),    # glint
    "n": (208, 164, 138),
    "m": (182, 104, 100),    # lips
    "t": (246, 246, 246),
    "W": (238, 238, 232),    # white top
    "b": (150, 16, 16),
    "B": (216, 34, 34),
    "x": (110, 86, 72),
}


class Shades(AsciiFace):
    """Long wavy hair, big round sunglasses, half-smile."""

    name = "shades"
    palette = SHADES_PALETTE
    shut = "w"

    base = [
        "....HHHHHHHH....",
        "..HHHHHHHHHHHH..",
        ".HHHHhhhhhhHHHH.",
        ".HHHhssssssHHHH.",
        ".HHhssssssssHHH.",
        ".HHhggggggggghH.",
        ".HgwwwwggwwwwgH.",
        ".HgwwwwggwwwwgH.",
        ".HhssssnnsssshH.",
        ".HHsssssnnssssHH",
        ".HHsssssssssssH.",
        ".HHsssmmmmsssHH.",
        ".HHHssssssssHHH.",
        ".HHHHssssssHHHH.",
        ".HHHHWWWWWWHHHH.",
        ".HHHWWWWWWWWHHH.",
    ]

    eye_rows = (6, 7)
    eyes = ((3, 6), (9, 12))
    mouth_row = 11

    mouths = {
        "set":   ".HHsssmmmmsssHH.",
        "calm":  ".HHssmmmmmmssHH.",
        "grin":  ".HHsmttttttmsHH.",
        "pain":  ".HHsmmmmmmmmsHH.",
        "dead":  ".HHsmmmmmmmmsHH.",
    }

    blood = [
        (6, 4), (5, 4), (6, 3),
        (10, 4), (9, 4), (10, 3),
        (3, 9), (2, 10), (3, 10),
        (12, 9), (13, 10), (12, 10),
        (4, 11), (11, 11), (5, 12), (10, 12),
        (7, 2), (8, 2), (2, 8), (13, 8),
    ]


class Face:
    """Idle mugshot: glances around and reacts to whatever you feed it.

    Doom drove the face from the player's state, so this keeps state rather than
    being a function of time — health persists, and a hit shows for a moment
    before the face settles back.
    """

    name = "face"

    def __init__(self, seed: int = 0, character: Character | None = None,
                 health: int = 100, fps: float = 20, feed=None, pace: float = 1.0):
        self.rng = random.Random(seed)
        self.character = character or CHARACTERS["doom"]
        self.health = health
        self.fps = fps
        self.feed = feed
        self.pace = pace

        self.gaze = 0
        self.mood = "idle"
        self.blink_left = 0
        self.mood_left = 0
        self.gaze_left = self._beats(self.rng.uniform(0.7, 2.2) * pace)

    def _beats(self, seconds: float) -> int:
        return max(1, round(seconds * self.fps))

    def trigger(self, event: str, amount: int = 20) -> None:
        if event == "hurt":
            self.health = max(0, self.health - amount)
            self.mood, self.mood_left = "ouch", self._beats(0.7)
            self.gaze, self.gaze_left = 0, self._beats(0.7)
        elif event == "heal":
            self.health = min(100, self.health + amount)
        elif event == "health":
            self.health = max(0, min(100, amount))
        elif event in ("evil", "kill", "god"):
            self.mood, self.mood_left = event, self._beats(2.0)
        elif event == "revive":
            self.health = 100
            self.mood, self.mood_left = "idle", 0

    def next(self) -> list[RGB]:
        if self.feed is not None:
            for event, amount in self.feed.drain():
                self.trigger(event, amount)

        if self.mood_left:
            self.mood_left -= 1
            if not self.mood_left:
                self.mood = "idle"

        if self.health > 0:
            self.gaze_left -= 1
            if self.gaze_left <= 0:
                # Looking straight ahead reads as attentive, so favour it.
                self.gaze = self.rng.choice([-1, 0, 0, 1])
                self.gaze_left = self._beats(self.rng.uniform(0.6, 2.4) * self.pace)

            if self.blink_left:
                self.blink_left -= 1
            elif self.rng.random() < 0.7 / (self.fps * self.pace):
                self.blink_left = self._beats(0.12)

        return self.character.frame(self.health, self.gaze,
                                    blink=bool(self.blink_left), mood=self.mood)


CHARACTERS: dict[str, Character] = {
    c.name: c for c in (DoomGuy(), SpriteFace("doomsprite", "doom"), MaxDamage(), Shades())
}
