"""Timebox Evo wire protocol.

Framing and pixel packing follow RomRider/node-divoom-timebox-evo PROTOCOL.md,
cross-checked against d03n3rfr1tz3/hass-divoom and spezifisch/divo.
"""

import struct

SCREEN = 16
PIXELS = SCREEN * SCREEN

# The device drops anything larger in a single animation packet.
CHUNK = 200

# Animations must be paced. Written back to back, the Mac's Bluetooth stack
# queues chunks faster than the Evo consumes them; it silently discards the tail
# and falls back to the clock. Measured against the panel: 0ms fails every time
# past ~80 chunks, 5ms never does.
CHUNK_GAP = 0.010

CMD_SET_IMAGE = 0x44
CMD_SET_VIEW = 0x45
CMD_GET_VIEW = 0x46
CMD_SET_ANIMATION_FRAME = 0x49
CMD_SET_TOOL = 0x72
CMD_SET_BRIGHTNESS = 0x74
CMD_SET_TIME_TYPE = 0x2D
CMD_SET_DATE_TIME = 0x18
CMD_SET_VOLUME = 0x08
CMD_SET_TEMP = 0x5F

RGB = tuple[int, int, int]


# hass-divoom widens the checksum to 32 bits once the byte sum passes 65535,
# where node-divoom-timebox-evo and divo stay at 16. Moot in practice: capping
# the palette at 255 keeps every frame under that sum. Kept as an escape hatch.
WIDE_CRC = False


def message(cmd: int, args: bytes = b"") -> bytes:
    """01 <len> <cmd> <args> <crc> 02, everything little-endian."""
    payload = struct.pack("<H", len(args) + 3) + bytes([cmd]) + bytes(args)
    total = sum(payload)
    crc = struct.pack("<I", total) if WIDE_CRC and total >= 0xFFFF else struct.pack("<H", total % 0x10000)
    return b"\x01" + payload + crc + b"\x02"


def pack_pixels(indices: list[int], palette_size: int) -> bytes:
    """Palette indices, LSB-first, ceil(log2(n)) bits each, spilling across bytes."""
    bits = max(1, (palette_size - 1).bit_length())
    mask = (1 << bits) - 1
    out = bytearray()
    acc = nbits = 0
    for index in indices:
        acc |= (index & mask) << nbits
        nbits += bits
        while nbits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            nbits -= 8
    if nbits:
        out.append(acc & 0xFF)
    return bytes(out)


def quantize(pixels: list[RGB]) -> tuple[list[RGB], list[int]]:
    """Build the frame palette in first-seen order."""
    palette: list[RGB] = []
    index_of: dict[RGB, int] = {}
    indices = []
    for px in pixels:
        i = index_of.get(px)
        if i is None:
            i = len(palette)
            index_of[px] = i
            palette.append(px)
        indices.append(i)
    return palette, indices


MAX_COLORS = 255


def frame_body(pixels: list[RGB], duration_ms: int = 0, animated: bool = False) -> bytes:
    palette, indices = quantize(pixels)
    # PROTOCOL.md says a count of 0 means 256 colours. It does not: the Evo
    # renders garbage for that one value and is fine at 255. Callers must
    # reduce beforehand — see canvas.reduce_palette.
    if len(palette) > MAX_COLORS:
        raise ValueError(f"{len(palette)} colours; the Evo only decodes up to {MAX_COLORS}")
    count = len(palette)
    body = struct.pack("<H", duration_ms if animated else 0)
    body += b"\x00"  # palette reset flag
    body += bytes([count])
    for r, g, b in palette:
        body += bytes([r, g, b])
    body += pack_pixels(indices, len(palette))
    return body


def frame(pixels: list[RGB], duration_ms: int = 0, animated: bool = False) -> bytes:
    body = frame_body(pixels, duration_ms, animated)
    return b"\xaa" + struct.pack("<H", len(body) + 3) + body


def image_messages(pixels: list[RGB]) -> list[bytes]:
    """A single still frame, sent whole under the fixed 000A0A04 header."""
    return [message(CMD_SET_IMAGE, b"\x00\x0a\x0a\x04" + frame(pixels))]


def animation_messages(frames: list[tuple[list[RGB], int]]) -> list[bytes]:
    """Frames are concatenated, then sliced into indexed chunks."""
    blob = b""
    for pixels, duration_ms in frames:
        blob += frame(pixels, duration_ms, animated=True)

    count = (len(blob) + CHUNK - 1) // CHUNK
    if count > 256:
        raise ValueError(
            f"{len(blob)} bytes needs {count} chunks; the index field is one byte, "
            "so it would wrap and corrupt the animation. Use fewer frames or colours."
        )

    out = []
    for index in range(count):
        part = blob[index * CHUNK:(index + 1) * CHUNK]
        args = struct.pack("<H", len(blob)) + bytes([index]) + part
        out.append(message(CMD_SET_ANIMATION_FRAME, args))
    return out


def brightness(value: int) -> bytes:
    return message(CMD_SET_BRIGHTNESS, bytes([max(0, min(100, value))]))


def light(color: RGB, level: int = 100, power: bool = True, effect: int = 0) -> bytes:
    r, g, b = color
    args = bytes([0x01, r, g, b, max(0, min(100, level)), effect, 0x01 if power else 0x00, 0, 0, 0])
    return message(CMD_SET_VIEW, args)


def clock(style: int = 0, twentyfour: bool = True, weather: bool = False,
          temp: bool = False, calendar: bool = False, color: RGB = (255, 255, 255)) -> bytes:
    args = bytes([
        0x00,
        0x01 if twentyfour else 0x00,
        style & 0xFF,
        0x01,
        0x01 if weather else 0x00,
        0x01 if temp else 0x00,
        0x01 if calendar else 0x00,
        *color,
    ])
    return message(CMD_SET_VIEW, args)


def scoreboard(red: int, blue: int) -> bytes:
    return message(CMD_SET_TOOL, b"\x01\x01" + struct.pack("<HH", red, blue))


def get_view() -> bytes:
    return message(CMD_GET_VIEW)
