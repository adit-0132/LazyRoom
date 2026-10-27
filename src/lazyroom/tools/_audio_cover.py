"""Minimal embedded-cover-art readers (MP3 ID3v2, FLAC, MP4/M4A, Ogg Vorbis/Opus).

Written from the format specifications (no mutagen, which is GPL):

* ID3v2.2 ``PIC`` / ID3v2.3-2.4 ``APIC`` frames, tag-level and (v2.4) frame-level
  unsynchronisation, extended headers, v2.3/v2.4 zlib-compressed frames and the v2.4
  data-length indicator.  Encrypted frames are skipped.
* FLAC ``METADATA_BLOCK_PICTURE`` (block type 6), also after a leading ID3v2 tag.
* MP4/M4A/M4B ``moov/udta/meta/ilst/covr/data`` atoms (ISO base media file format,
  32/64-bit box sizes, full-box and QuickTime-style ``meta``).
* Ogg Vorbis (``\\x03vorbis``) and Opus (``OpusTags``) comment packets:
  ``METADATA_BLOCK_PICTURE`` (base64 FLAC picture) and legacy ``COVERART``.

Every reader returns a list of ``Picture``; nothing is decoded here.
"""
from __future__ import annotations

import base64
import binascii
import mmap
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

PICTURE_TYPES = ["other", "file_icon", "other_file_icon", "front_cover", "back_cover", "leaflet", "media",
                 "lead_artist", "artist", "conductor", "band", "composer", "lyricist", "recording_location",
                 "during_recording", "during_performance", "screen_capture", "bright_fish", "illustration",
                 "band_logotype", "publisher_logotype"]
MAX_PICTURE_BYTES = 64 * 1024 * 1024


@dataclass
class Picture:
    data: bytes
    mime: str = ""
    picture_type: int = 0
    description: str = ""
    container: str = ""
    extra: dict = field(default_factory=dict)

    def describe(self) -> dict:
        return {"mime": self.mime or sniff_mime(self.data), "type": PICTURE_TYPES[self.picture_type]
                if 0 <= self.picture_type < len(PICTURE_TYPES) else str(self.picture_type),
                "description": self.description, "bytes": len(self.data), "container": self.container}


def sniff_mime(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:2] == b"BM":
        return "image/bmp"
    return "application/octet-stream"


class FormatError(ValueError):
    pass


# ---------------------------------------------------------------------------
# ID3v2
# ---------------------------------------------------------------------------

def _syncsafe(b: bytes) -> int:
    return (b[0] & 0x7F) << 21 | (b[1] & 0x7F) << 14 | (b[2] & 0x7F) << 7 | (b[3] & 0x7F)


def _unsync(b: bytes) -> bytes:
    return b.replace(b"\xff\x00", b"\xff")


def _split_text(b: bytes, enc: int) -> tuple[str, bytes]:
    """Split an encoded, terminated string from the front of ``b``."""
    if enc in (1, 2):  # UTF-16 (BOM) / UTF-16BE: two-byte aligned terminator
        i = 0
        while i + 1 < len(b) and b[i:i + 2] != b"\x00\x00":
            i += 2
        raw, rest = b[:i], b[i + 2:]
        try:
            text = raw.decode("utf-16" if enc == 1 else "utf-16-be")
        except UnicodeDecodeError:
            text = ""
        return text, rest
    i = b.find(b"\x00")
    if i < 0:
        i = len(b)
    raw, rest = b[:i], b[i + 1:]
    return raw.decode("utf-8" if enc == 3 else "latin-1", errors="replace"), rest


def id3_tag_size(buf: bytes | mmap.mmap, offset: int = 0) -> int:
    """Total size of an ID3v2 tag at ``offset`` (0 if none)."""
    h = buf[offset:offset + 10]
    if len(h) < 10 or h[:3] != b"ID3" or h[3] == 0xFF or any(x & 0x80 for x in h[6:10]):
        return 0
    return 10 + _syncsafe(h[6:10]) + (10 if h[3] == 4 and h[5] & 0x10 else 0)


def parse_id3(buf: bytes | mmap.mmap, offset: int = 0) -> list[Picture]:
    h = buf[offset:offset + 10]
    if len(h) < 10 or h[:3] != b"ID3":
        return []
    major, flags = h[3], h[5]
    if major not in (2, 3, 4):
        raise FormatError(f"unsupported ID3v2.{major}")
    size = _syncsafe(h[6:10])
    body = bytes(buf[offset + 10:offset + 10 + size])
    if flags & 0x80 and major < 4:  # whole-tag unsynchronisation (v2.4 marks frames individually)
        body = _unsync(body)
    pos = 0
    if flags & 0x40 and major == 3:
        pos = 4 + struct.unpack(">I", body[:4])[0]
    elif flags & 0x40 and major == 4:
        pos = _syncsafe(body[:4])
    pics: list[Picture] = []
    while pos < len(body):
        if major == 2:
            if pos + 6 > len(body) or body[pos] == 0:
                break
            fid, fsize, fflags = body[pos:pos + 3], int.from_bytes(body[pos + 3:pos + 6], "big"), 0
            pos += 6
        else:
            if pos + 10 > len(body) or body[pos] == 0:
                break
            fid = body[pos:pos + 4]
            raw_size = body[pos + 4:pos + 8]
            fsize = _syncsafe(raw_size) if major == 4 else struct.unpack(">I", raw_size)[0]
            fflags = struct.unpack(">H", body[pos + 8:pos + 10])[0]
            pos += 10
        data = body[pos:pos + fsize]
        pos += fsize
        if fid not in (b"APIC", b"PIC"):
            continue
        data = _id3_frame_payload(data, major, fflags, bool(flags & 0x80))
        if data is None or len(data) < 4:
            continue
        pic = _parse_apic(data, major)
        if pic is not None:
            pics.append(pic)
    return pics


def _id3_frame_payload(data: bytes, major: int, fflags: int, tag_unsync: bool) -> bytes | None:
    if major == 3:
        if fflags & 0x0040:  # encrypted
            return None
        extra = (4 if fflags & 0x0080 else 0) + (1 if fflags & 0x0020 else 0)
        compressed = bool(fflags & 0x0080)
        data = data[extra:]
    elif major == 4:
        if fflags & 0x0004:  # encrypted
            return None
        if fflags & 0x0002 or tag_unsync:  # covers everything after the frame header
            data = _unsync(data)
        if fflags & 0x0040:  # grouping identity
            data = data[1:]
        if fflags & 0x0001:  # data length indicator
            data = data[4:]
        compressed = bool(fflags & 0x0008)
    else:
        compressed = False
    if compressed:
        try:
            data = zlib.decompressobj().decompress(data, MAX_PICTURE_BYTES)
        except zlib.error:
            return None
    return data


def _parse_apic(data: bytes, major: int) -> Picture | None:
    enc = data[0]
    if major == 2:  # PIC: 3-char image format
        fmt = data[1:4].decode("latin-1", errors="replace").upper()
        mime = {"JPG": "image/jpeg", "PNG": "image/png", "GIF": "image/gif", "BMP": "image/bmp"}.get(fmt, "")
        ptype = data[4]
        rest = data[5:]
    else:
        i = data.find(b"\x00", 1)
        if i < 0:
            return None
        mime = data[1:i].decode("latin-1", errors="replace").lower()
        if mime and "/" not in mime:
            mime = "image/" + {"jpg": "jpeg"}.get(mime, mime)
        ptype = data[i + 1] if i + 1 < len(data) else 0
        rest = data[i + 2:]
    desc, img = _split_text(rest, enc)
    if not img:
        return None
    return Picture(img, mime, ptype, desc, f"id3v2.{major}")


# ---------------------------------------------------------------------------
# FLAC picture block (also used by Ogg METADATA_BLOCK_PICTURE)
# ---------------------------------------------------------------------------

def parse_flac_picture(b: bytes, container: str = "flac") -> Picture:
    try:
        ptype, mlen = struct.unpack_from(">II", b, 0)
        p = 8
        mime = b[p:p + mlen].decode("ascii", errors="replace")
        p += mlen
        (dlen,) = struct.unpack_from(">I", b, p)
        p += 4
        desc = b[p:p + dlen].decode("utf-8", errors="replace")
        p += dlen
        w, h, depth, ncol, size = struct.unpack_from(">IIIII", b, p)
        p += 20
    except struct.error as exc:
        raise FormatError(f"truncated picture block: {exc}") from None
    data = b[p:p + size]
    if len(data) != size:
        raise FormatError("truncated picture data")
    return Picture(data, mime.lower(), ptype, desc, container, {"width": w, "height": h, "depth": depth})


def parse_flac(buf: bytes | mmap.mmap) -> list[Picture]:
    pos = id3_tag_size(buf, 0)
    if buf[pos:pos + 4] != b"fLaC":
        raise FormatError("not a FLAC stream")
    pos += 4
    pics = []
    while pos + 4 <= len(buf):
        hdr = buf[pos]
        last, btype = hdr & 0x80, hdr & 0x7F
        blen = int.from_bytes(buf[pos + 1:pos + 4], "big")
        pos += 4
        if btype == 6:
            pics.append(parse_flac_picture(bytes(buf[pos:pos + blen])))
        elif btype == 127:
            raise FormatError("invalid metadata block type")
        pos += blen
        if last:
            break
    return pics


# ---------------------------------------------------------------------------
# MP4 / ISO BMFF
# ---------------------------------------------------------------------------

def _atoms(buf: bytes | mmap.mmap, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[pos:pos + 8])
        hdr = 8
        if size == 1:
            if pos + 16 > end:
                return
            size = struct.unpack(">Q", buf[pos + 8:pos + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - pos
        if size < hdr or pos + size > end:
            return
        yield typ, pos + hdr, pos + size
        pos += size


def _child(buf, start: int, end: int, name: bytes) -> tuple[int, int] | None:
    for typ, a, b in _atoms(buf, start, end):
        if typ == name:
            return a, b
    return None


def parse_mp4(buf: bytes | mmap.mmap) -> list[Picture]:
    end = len(buf)
    if buf[4:8] != b"ftyp" and _child(buf, 0, end, b"moov") is None:
        raise FormatError("not an MP4/M4A file")
    moov = _child(buf, 0, end, b"moov")
    if moov is None:
        return []
    pics: list[Picture] = []
    metas = []
    udta = _child(buf, *moov, b"udta")
    if udta:
        metas.append(_child(buf, *udta, b"meta"))
    metas.append(_child(buf, *moov, b"meta"))
    for meta in metas:
        if not meta:
            continue
        a, b = meta
        # ISO full box (4 bytes version/flags) vs QuickTime-style meta (children start immediately)
        if buf[a + 4:a + 8] != b"hdlr" or int.from_bytes(buf[a:a + 4], "big") < 8:
            a += 4
        ilst = _child(buf, a, b, b"ilst")
        if not ilst:
            continue
        covr = _child(buf, *ilst, b"covr")
        if not covr:
            continue
        for typ, da, db in _atoms(buf, *covr):
            if typ != b"data" or db - da < 8:
                continue
            kind = int.from_bytes(buf[da + 1:da + 4], "big")
            data = bytes(buf[da + 8:db])
            mime = {13: "image/jpeg", 14: "image/png", 27: "image/bmp"}.get(kind, "")
            pics.append(Picture(data, mime, 3, "", "mp4", {"data_type": kind}))
        if pics:
            break
    return pics


# ---------------------------------------------------------------------------
# Ogg (Vorbis / Opus comment header)
# ---------------------------------------------------------------------------

def _ogg_packets(buf: bytes | mmap.mmap, limit: int = 3, max_bytes: int = 64 * 1024 * 1024):
    """Packets of the first logical bitstream (only the first ``limit`` packets)."""
    pos, serial, packet, count, total = 0, None, bytearray(), 0, 0
    while pos + 27 <= len(buf) and count < limit:
        if buf[pos:pos + 4] != b"OggS":
            raise FormatError("lost Ogg page sync")
        nseg = buf[pos + 26]
        sid = struct.unpack("<I", buf[pos + 14:pos + 18])[0]
        segs = buf[pos + 27:pos + 27 + nseg]
        p = pos + 27 + nseg
        if serial is None:
            serial = sid
        for lace in segs:
            if sid == serial:
                packet += buf[p:p + lace]
                total += lace
                if total > max_bytes:
                    raise FormatError("Ogg header packet too large")
                if lace < 255:
                    yield bytes(packet)
                    packet = bytearray()
                    count += 1
                    if count >= limit:
                        return
            p += lace
        pos = p


def parse_vorbis_comments(c: bytes, container: str) -> list[Picture]:
    try:
        vlen = struct.unpack_from("<I", c, 0)[0]
        p = 4 + vlen
        n = struct.unpack_from("<I", c, p)[0]
        p += 4
        pics = []
        for _ in range(n):
            ln = struct.unpack_from("<I", c, p)[0]
            p += 4
            entry = c[p:p + ln]
            p += ln
            key, _, value = entry.partition(b"=")
            k = key.upper()
            if k == b"METADATA_BLOCK_PICTURE":
                try:
                    pics.append(parse_flac_picture(base64.b64decode(value, validate=False), container))
                except (binascii.Error, FormatError):
                    continue
            elif k == b"COVERART":
                try:
                    data = base64.b64decode(value, validate=False)
                except binascii.Error:
                    continue
                pics.append(Picture(data, "", 3, "", container + " (COVERART)"))
        return pics
    except struct.error:
        raise FormatError("truncated comment header") from None


def parse_ogg(buf: bytes | mmap.mmap) -> list[Picture]:
    packets = list(_ogg_packets(buf))
    if not packets:
        raise FormatError("empty Ogg stream")
    first = packets[0]
    if first[:7] == b"\x01vorbis":
        com = next((p for p in packets[1:] if p[:7] == b"\x03vorbis"), None)
        return parse_vorbis_comments(com[7:], "ogg-vorbis") if com else []
    if first[:8] == b"OpusHead":
        com = next((p for p in packets[1:] if p[:8] == b"OpusTags"), None)
        return parse_vorbis_comments(com[8:], "ogg-opus") if com else []
    if first[:5] == b"\x7fFLAC":  # Ogg FLAC: metadata blocks follow as packets
        pics = []
        for p in packets[1:]:
            if p and p[0] & 0x7F == 6:
                pics.append(parse_flac_picture(p[4:], "ogg-flac"))
        return pics
    raise FormatError("unsupported Ogg codec")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def detect(buf: bytes | mmap.mmap) -> str:
    head = bytes(buf[:12])
    if head[:4] == b"fLaC":
        return "flac"
    if head[:3] == b"ID3":
        n = id3_tag_size(buf)
        return "flac" if buf[n:n + 4] == b"fLaC" else "mp3"
    if head[:4] == b"OggS":
        return "ogg"
    if head[4:8] in (b"ftyp", b"moov", b"mdat", b"free", b"wide", b"skip"):
        return "mp4"
    if len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0:
        return "mp3"  # MPEG audio without an ID3v2 tag
    return "unknown"


def read_pictures(path: str | Path) -> tuple[str, list[Picture]]:
    p = Path(path)
    if p.stat().st_size == 0:
        return "unknown", []
    with open(p, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf:
        kind = detect(buf)
        if kind == "mp3":
            return kind, parse_id3(buf)
        if kind == "flac":
            return kind, parse_flac(buf)
        if kind == "mp4":
            return kind, parse_mp4(buf)
        if kind == "ogg":
            return kind, parse_ogg(buf)
        return kind, []


def choose(pics: list[Picture], index: int | None) -> Picture | None:
    """Explicit index, else the first front cover, else the first picture."""
    if not pics:
        return None
    if index is not None:
        return pics[index]
    return next((p for p in pics if p.picture_type == 3), pics[0])
