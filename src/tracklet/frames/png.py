"""PNG encoding with nothing but the standard library.

``archive`` is stdlib-only so it runs in a bare container; the bake step that
feeds the web tool has the same constraint, and it needs to write images. PNG is
a short enough format to write directly: an 8-byte signature and a sequence of
length-tag-data-CRC chunks, with the pixel data zlib-deflated after a per-row
filter byte. ``zlib`` and ``struct`` cover all of it, so there is no Pillow
dependency and no numpy in the image path.

Two encodings are used:

``write_gray``
    8-bit greyscale for the sky previews. Filter type 1 (``Sub``) is applied
    because astronomical backgrounds are smooth, and differencing along the row
    shrinks a typical SPHEREx preview by a further third over no filtering.

``write_rgb`` / colour maps
    The terminal UI wants a phosphor ramp rather than grey. Palettes live here so
    the Python bake and the JavaScript renderer can be driven from one
    definition, exported by :func:`palette_stops`.

:func:`read_png` goes the other way. It exists because the all-sky texture the
globe draws is not ours to generate -- it is a HiPS tile published by CDS, and
vendoring it means decoding someone else's encoder output, which uses filters
this module never emits.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Iterable, Sequence

PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _filter_sub(row: bytes, stride: int) -> bytes:
    """PNG filter type 1: each byte minus the byte ``stride`` to its left."""
    out = bytearray(row)
    for i in range(len(row) - 1, stride - 1, -1):
        out[i] = (row[i] - row[i - stride]) & 0xFF
    return bytes(out)


def _encode(
    width: int,
    height: int,
    rows: Iterable[bytes],
    *,
    color_type: int,
    stride: int,
    level: int = 9,
    extra: bytes = b"",
    filter_type: int = 1,
    bit_depth: int = 8,
) -> bytes:
    """Encode rows already packed to ``stride`` bytes per pixel.

    ``filter_type`` is 1 (``Sub``) or 0 (``None``). ``bit_depth`` is 8 or 16;
    it only reaches the IHDR, since ``rows`` arrive already packed and
    ``stride`` is what the filter works in.

    Sub was once said to shrink "smooth astronomical backgrounds" by about a
    third. It does not, on anything this package writes. Measured against the
    files on disk, by decoding each one and re-encoding it both ways:

    * **The 30 baked 16-bit stamps: Sub *costs* a median 4.1%** (2.2% to 10.6%)
      and helps on none of them. Re-encoding with filter ``None`` reproduces
      the on-disk bytes exactly for all 30, so this is a real comparison and
      not a decoding artefact -- and it is also how we know the stamps that ship
      are unfiltered.
    * **The all-sky grey+alpha: Sub costs 7.3%** on the sheet currently baked --
      1728x1856 at native 64px tiles, 1,983.9 KB becoming 2,128.3 KB, again
      reproducing the shipped bytes with ``None``.

    Sub differences along the row, which pays only where neighbouring pixels
    correlate more strongly than the entropy coder already exploits. A
    background-limited SPHEREx stamp is mostly independent per-pixel noise:
    nothing for the filter to win, and a filter-type byte per row to lose.

    Both figures have moved before, and the reason is worth keeping: this note
    read 25% for the all-sky, was "corrected" to 3.9% while ``bake_sky.py`` was
    downsampling 2x before the encoder saw the tiles, and went back to ~25% when
    that default returned to 1. Then the sheet was re-baked with a different
    rectification and the *file* changed size, which moved the percentage again.
    A figure like this belongs to a specific file, so it is quoted with the
    dimensions and the kilobytes rather than on its own.

    ``write_png16`` and ``write_gray_alpha`` are called with ``filter_type=0``
    everywhere it matters, and both shipped sets re-encode to their own bytes
    that way. ``tests/test_png.py`` pins the direction so the conclusion cannot
    invert without a test noticing.
    """
    if filter_type not in (0, 1):
        raise ValueError(f"filter_type {filter_type} unsupported, use 0 or 1")
    if bit_depth not in (8, 16):
        raise ValueError(f"bit_depth {bit_depth} unsupported, use 8 or 16")
    raw = bytearray()
    for row in rows:
        raw.append(filter_type)
        raw += _filter_sub(row, stride) if filter_type else row
    ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    return (
        PNG_SIG
        + _chunk(b"IHDR", ihdr)
        + extra
        + _chunk(b"IDAT", zlib.compress(bytes(raw), level))
        + _chunk(b"IEND", b"")
    )


def write_gray(levels: Sequence[Sequence[int]], *, level: int = 9) -> bytes:
    """8-bit greyscale PNG from rows of 0-255 ints."""
    h = len(levels)
    w = len(levels[0]) if h else 0
    return _encode(
        w,
        h,
        (bytes(max(0, min(255, int(v))) for v in row) for row in levels),
        color_type=0,
        stride=1,
        level=level,
    )


def write_gray_alpha(
    levels: Sequence[Sequence[int]],
    alpha: Sequence[Sequence[int]],
    *,
    level: int = 9,
    filter_type: int = 1,
) -> bytes:
    """Greyscale + alpha, so masked (NaN) pixels can be transparent.

    This is why the stamps are not plain greyscale: a NaN painted as black is
    indistinguishable from real dark sky, and a viewer blinking two epochs will
    read a detector artefact as a moving object. Transparent is unambiguous.
    """
    h = len(levels)
    w = len(levels[0]) if h else 0

    def rows():
        for lr, ar in zip(levels, alpha):
            out = bytearray()
            for v, a in zip(lr, ar):
                out.append(max(0, min(255, int(v))))
                out.append(max(0, min(255, int(a))))
            yield bytes(out)

    return _encode(
        w, h, rows(), color_type=4, stride=2, level=level, filter_type=filter_type
    )


def write_rgb(pixels: Sequence[Sequence[tuple[int, int, int]]], *, level: int = 9) -> bytes:
    """24-bit RGB PNG."""
    h = len(pixels)
    w = len(pixels[0]) if h else 0

    def rows():
        for row in pixels:
            out = bytearray()
            for r, g, b in row:
                out += bytes(
                    (
                        max(0, min(255, int(r))),
                        max(0, min(255, int(g))),
                        max(0, min(255, int(b))),
                    )
                )
            yield bytes(out)

    return _encode(w, h, rows(), color_type=2, stride=3, level=level)


def write_png16(
    values: Sequence[Sequence[int]],
    *,
    level: int = 9,
    filter_type: int = 0,
) -> bytes:
    """16-bit greyscale, big-endian: a lossless carrier for scaled float data.

    A display PNG throws away 8 bits, which is fine for looking at but not for
    measuring. When the bake step wants to ship *data* rather than a picture --
    so the browser can report a real surface brightness under the cursor, and
    re-stretch without a round trip -- it quantises to 16 bits over a recorded
    ``[vmin, vmax]`` and ships that. 65535 levels across a zscale range is finer
    than the per-pixel noise by a wide margin, so nothing measurable is lost.

    ``filter_type`` defaults to 0 (``None``). This used to hardcode 1 (``Sub``),
    which measures a median **4.1% larger** across the 30 baked stamps and wins
    on none of them -- see :func:`_encode` for why row differencing has nothing
    to exploit in background-limited noise. The stamps on disk have since been
    re-baked and all 30 re-encode to their own bytes with ``None``.
    ``web/js/fits.js::decodePng16`` unfilters types 0-4, so the browser reads
    either without changing.
    """
    h = len(values)
    w = len(values[0]) if h else 0
    rows = (
        b"".join(struct.pack(">H", max(0, min(65535, int(v)))) for v in row)
        for row in values
    )
    return _encode(
        w, h, rows, color_type=0, stride=2, level=level,
        filter_type=filter_type, bit_depth=16,
    )


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


class PngError(ValueError):
    """A PNG this module declines to decode."""


#: Bytes per pixel for the colour types :func:`read_png` accepts.
_CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}


def _unfilter(raw: bytes, width: int, height: int, bpp: int) -> bytearray:
    """Undo the per-row filter. Types 0-4, the whole of the PNG spec.

    The writers above only ever emit ``Sub``, so this is the one function here
    that has to handle input it did not produce -- an all-sky tile from CDS uses
    ``Paeth`` on most rows. Filters reference the pixel to the left and the row
    above, so this is inherently sequential and stays in a ``bytearray``.
    """
    stride = width * bpp
    out = bytearray(height * stride)
    prev = bytearray(stride)
    pos = 0
    for y in range(height):
        if pos >= len(raw):
            raise PngError(f"truncated image data at row {y} of {height}")
        ftype = raw[pos]
        pos += 1
        line = bytearray(raw[pos : pos + stride])
        if len(line) < stride:
            raise PngError(f"truncated scanline at row {y}")
        pos += stride
        if ftype == 1:  # Sub
            for x in range(bpp, stride):
                line[x] = (line[x] + line[x - bpp]) & 0xFF
        elif ftype == 2:  # Up
            for x in range(stride):
                line[x] = (line[x] + prev[x]) & 0xFF
        elif ftype == 3:  # Average
            for x in range(stride):
                left = line[x - bpp] if x >= bpp else 0
                line[x] = (line[x] + ((left + prev[x]) >> 1)) & 0xFF
        elif ftype == 4:  # Paeth
            for x in range(stride):
                a = line[x - bpp] if x >= bpp else 0
                b = prev[x]
                c = prev[x - bpp] if x >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[x] = (line[x] + pred) & 0xFF
        elif ftype != 0:
            raise PngError(f"unknown filter type {ftype} at row {y}")
        out[y * stride : (y + 1) * stride] = line
        prev = line
    return out


def read_png16(data: bytes) -> list[list[int]]:
    """Decode a 16-bit greyscale PNG to rows of ``0..65535``.

    The inverse of :func:`write_png16`, and it exists because without it the
    bake could write data nothing on the Python side could read back. That
    asymmetry is not merely inelegant: it meant the only way to check what the
    bake had actually produced was to open the browser, which is how a pixel
    pipeline change gets shipped without anyone re-measuring the numbers that
    depend on it. ``scripts/measure_search.py`` reads stamps through this.

    Mirrors ``decodePng16`` in web/js/fits.js. Index 0 is the NaN sentinel the
    bake reserves; callers map it back themselves, since what "missing" means
    in physical units is theirs to decide.

    Deliberately narrow, like :func:`read_png`: greyscale, 16 bits, no
    interlace. Anything else is an error rather than a guess.
    """
    if data[:8] != PNG_SIG:
        raise PngError("not a PNG")
    width = height = bitdepth = color = interlace = -1
    idat = bytearray()
    pos = 8
    seen_ihdr = False
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        if tag == b"IHDR":
            width, height, bitdepth, color, _comp, _filt, interlace = struct.unpack(
                ">IIBBBBB", body[:13]
            )
            seen_ihdr = True
        elif tag == b"IDAT":
            idat += body
        elif tag == b"IEND":
            break
        pos += 12 + length

    if not seen_ihdr:
        raise PngError("no IHDR chunk")
    if bitdepth != 16:
        raise PngError(f"bit depth {bitdepth} unsupported here, only 16")
    if color != 0:
        raise PngError(f"colour type {color} unsupported here, only greyscale")
    if interlace:
        raise PngError("interlaced PNG unsupported")
    if not idat:
        raise PngError("no image data")

    # Two bytes per pixel, so the filter's "pixel to the left" is two bytes
    # back -- passing that as the byte depth is what makes `_unfilter` correct
    # for 16-bit data rather than subtly wrong on every filtered row.
    flat = _unfilter(zlib.decompress(bytes(idat)), width, height, 2)
    out: list[list[int]] = []
    for j in range(height):
        base = j * width * 2
        out.append([
            (flat[base + 2 * i] << 8) | flat[base + 2 * i + 1] for i in range(width)
        ])
    return out


def read_png(data: bytes) -> tuple[int, int, int, bytearray]:
    """Decode an 8-bit PNG to ``(width, height, channels, pixels)``.

    Deliberately narrow: 8 bits per channel, no interlace, no palette. That
    covers what this project actually reads -- its own output, and the HiPS
    all-sky tiles ``scripts/bake_sky.py`` vendors -- and refusing everything else
    is better than half-supporting it. ``pixels`` is row-major and tightly
    packed, so pixel ``(x, y)`` channel ``c`` is at ``(y * width + x) * ch + c``.
    """
    if data[:8] != PNG_SIG:
        raise PngError("not a PNG")
    width = height = bitdepth = color = interlace = -1
    idat = bytearray()
    pos = 8
    seen_ihdr = False
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        if tag == b"IHDR":
            width, height, bitdepth, color, _comp, _filt, interlace = struct.unpack(
                ">IIBBBBB", body[:13]
            )
            seen_ihdr = True
        elif tag == b"IDAT":
            idat += body
        elif tag == b"IEND":
            break
        pos += 12 + length

    if not seen_ihdr:
        raise PngError("no IHDR chunk")
    if bitdepth != 8:
        raise PngError(f"bit depth {bitdepth} unsupported, only 8")
    if interlace:
        raise PngError("interlaced PNG unsupported")
    if color not in _CHANNELS:
        raise PngError(f"colour type {color} unsupported")
    if not idat:
        raise PngError("no image data")

    channels = _CHANNELS[color]
    pixels = _unfilter(zlib.decompress(bytes(idat)), width, height, channels)
    return width, height, channels, pixels


# --------------------------------------------------------------------------- #
# Palettes
# --------------------------------------------------------------------------- #

#: Display ramps, as ``(position, (r, g, b))`` stops interpolated in sRGB, for
#: any image this side renders. **The web tool does not read these** -- it has
#: its own copy in ``web/js/palettes.js``, because a palette is a display choice
#: and shipping it inside a data manifest would mean re-fetching megabytes from
#: NASA to change a colour. The two are kept identical; if they drift, the
#: consequence is cosmetic, since nothing measured passes through a palette.
#:
#: ``gray`` is the image's ramp, and the web tool's Colour option keeps its
#: brightness and adds only a hue per visit; ``residual`` draws difference
#: images. The low end is held near black across most of the range: a
#: background-limited SPHEREx frame is nearly all sky, so a ramp that reaches
#: mid-brightness by its midpoint paints the whole field one flat tone and
#: leaves sources nowhere to go.
PALETTES: dict[str, tuple[tuple[float, tuple[int, int, int]], ...]] = {
    #: Strictly neutral. The default.
    "gray": (
        (0.00, (0, 0, 0)),
        (0.34, (15, 15, 15)),
        (0.60, (74, 74, 74)),
        (0.82, (172, 172, 172)),
        (1.00, (250, 250, 250)),
    ),
    #: Diverging, for difference frames. Two hues are not decoration: brightness
    #: alone cannot distinguish "got brighter" from "got dimmer" on a map that is
    #: symmetric about zero. Both sides are heavily desaturated and the midpoint
    #: is black, so no-change reads as background.
    "residual": (
        (0.00, (150, 176, 196)),
        (0.34, (40, 56, 68)),
        (0.50, (0, 0, 0)),
        (0.66, (74, 58, 44)),
        (1.00, (208, 184, 156)),
    ),
}


def palette_lut(name: str, n: int = 256) -> list[tuple[int, int, int]]:
    """Expand a palette to an ``n``-entry lookup table."""
    stops = PALETTES[name]
    lut: list[tuple[int, int, int]] = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0.0
        lo = stops[0]
        hi = stops[-1]
        for k in range(len(stops) - 1):
            if stops[k][0] <= t <= stops[k + 1][0]:
                lo, hi = stops[k], stops[k + 1]
                break
        span = hi[0] - lo[0]
        f = 0.0 if span <= 0 else (t - lo[0]) / span
        lut.append(
            tuple(round(lo[1][c] + f * (hi[1][c] - lo[1][c])) for c in range(3))  # type: ignore[misc]
        )
    return lut


def palette_stops() -> dict[str, list[list]]:
    """Palettes in a JSON-friendly shape, for the browser to use verbatim."""
    return {k: [[p, list(c)] for p, c in v] for k, v in PALETTES.items()}
