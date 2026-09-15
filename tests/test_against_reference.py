"""Differential test: our encoder vs the two reference implementations in ref/.

These are the only implementations anyone has confirmed working against real
hardware, so matching them byte-for-byte is the closest thing to a spec we get.
Skipped when ref/ has not been cloned.
"""

import importlib.util
import random
import sys
import types
from pathlib import Path

import pytest

from divoom import proto

ROOT = Path(__file__).resolve().parent.parent
DIVO = ROOT / "ref" / "divo"
HASS = ROOT / "ref" / "hass-divoom" / "custom_components" / "divoom" / "devices"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        pytest.skip(f"{path} missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def evo_encoder():
    if not (DIVO / "divo" / "evo_encoder.py").exists():
        pytest.skip("ref/divo not cloned")
    return _load("ref_evo_encoder", DIVO / "divo" / "evo_encoder.py").EvoEncoder


@pytest.fixture(scope="module")
def timebox():
    if not (HASS / "timebox.py").exists():
        pytest.skip("ref/hass-divoom not cloned")
    # timebox.py does `from .divoom import Divoom`, so it needs a real parent
    # package; synthesise one rather than touching the cloned tree.
    parent = types.ModuleType("refdev")
    parent.__path__ = [str(HASS)]  # type: ignore[attr-defined]
    sys.modules["refdev"] = parent
    _load("refdev.divoom", HASS / "divoom.py")
    module = _load("refdev.timebox", HASS / "timebox.py")
    return module.Timebox(mac="00:00:00:00:00:00")


def random_pixels(seed: int, distinct: int) -> list[tuple[int, int, int]]:
    rng = random.Random(seed)
    palette = [(rng.randrange(256), rng.randrange(256), rng.randrange(256))
               for _ in range(distinct)]
    return [rng.choice(palette) for _ in range(proto.PIXELS)]


PALETTE_SIZES = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 63, 100, 127, 128, 200, 255, 256]


@pytest.mark.parametrize("distinct", PALETTE_SIZES)
def test_palette_and_pixels_match_divo(evo_encoder, distinct):
    pixels = random_pixels(distinct, distinct)
    packed = [(r << 16) + (g << 8) + b for r, g, b in pixels]
    expected = bytes.fromhex(evo_encoder.encode_colours(packed))

    palette, indices = proto.quantize(pixels)
    ours = bytes([len(palette) % 256])
    for r, g, b in palette:
        ours += bytes([r, g, b])
    ours += proto.pack_pixels(indices, len(palette))

    assert ours == expected


@pytest.fixture
def wide_crc():
    proto.WIDE_CRC = True
    yield
    proto.WIDE_CRC = False


@pytest.mark.parametrize("distinct", PALETTE_SIZES)
def test_still_image_message_matches_hass_divoom(timebox, wide_crc, distinct):
    pixels = random_pixels(distinct + 1000, distinct)

    palette, indices = proto.quantize(pixels)
    count = len(palette) if len(palette) < 256 else 0
    body = timebox.process_frame(indices, [list(c) for c in palette], count, 1, 0, False)
    frame, length = timebox.make_frame(body)
    args = timebox.make_framepart(length, -1, frame)
    expected = bytes(timebox.make_message(
        list((len(args) + 3).to_bytes(2, "little")) + [0x44] + list(args)
    ))

    assert proto.image_messages(pixels)[0] == expected


def test_narrow_crc_stays_16_bit_on_a_large_palette():
    """The Evo-specific reference never widens; guard against regressing to hass-divoom."""
    msg = proto.image_messages(random_pixels(7, 256))[0]
    declared = int.from_bytes(msg[1:3], "little")
    crc_size = len(msg) - 2 - declared
    assert sum(msg[1:1 + declared]) > 0xFFFF, "palette too small to exercise the overflow"
    assert crc_size == 2


def test_animation_chunks_match_hass_divoom(timebox):
    frames = [(random_pixels(i, 6), 100) for i in range(4)]

    parts, total = [], 0
    for pixels, duration in frames:
        palette, indices = proto.quantize(pixels)
        body = timebox.process_frame(indices, [list(c) for c in palette], len(palette), 4, duration, False)
        frame, length = timebox.make_frame(body)
        parts += frame
        total += length

    expected = []
    for index, chunk in enumerate(timebox.chunks(parts, timebox.chunksize)):
        args = timebox.make_framepart(total, index, chunk)
        expected.append(bytes(timebox.make_message(
            list((len(args) + 3).to_bytes(2, "little")) + [0x49] + list(args)
        )))

    assert proto.animation_messages(frames) == expected


def test_framing_worked_by_hand():
    # brightness 50: len = 1 arg + 3 = 0x0004, crc = 04+00+74+32 = 0x00aa.
    assert proto.brightness(50).hex() == "0104007432aa0002"
