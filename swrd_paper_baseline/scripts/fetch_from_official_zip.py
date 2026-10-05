"""Fetch selected members of the official SWRD archive without downloading all of it.

The release is one zip on Google Drive (folder 1LNUt101wufTBJpRAgrZAU1h-Tfx629wO, file
``SWXD_Data.zip``, 124,398,609,179 bytes). Google Drive answers HTTP range requests, and a zip file keeps its
table of contents (the "central directory") at the end, so we can:

  1. read the last few MB -> list every member with its offset and compressed size,
  2. download only the members whose path matches ``--include`` (regex), each as one range request,
  3. decompress (deflate) or copy (stored) them to ``--out-dir``, verifying the CRC-32 from the directory.

Use ``--list`` first to see the archive layout and the total size of what a pattern would fetch.
Nothing else is downloaded.

Run (on the box):
  uv run python scripts/fetch_from_official_zip.py --list                       # table of contents only
  uv run python scripts/fetch_from_official_zip.py --include 'crop_weld_images/T/2/' --out-dir ~/swrd_paper_baseline/data/official
"""

from __future__ import annotations

import argparse
import io
import re
import struct
import sys
import zipfile
import zlib
from collections import Counter
from pathlib import Path, PurePosixPath

import requests
from tqdm import tqdm

FILE_ID = "1Hc9_de5YAdXg46F-GfjcRKMEBRF9XRQi"
URL = f"https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t"
TOTAL_SIZE = 124_398_609_179  # verified 2026-10-02 via Content-Range


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP range requests, with a read-ahead buffer.

    ``zipfile.ZipFile`` only needs seek/tell/read to parse the central directory; it issues a handful of
    reads near the end of the file, which the buffer turns into one or two requests.
    """

    def __init__(self, url: str, size: int, session: requests.Session, min_request: int = 8 << 20):
        self.url, self.size, self.session, self.min_request = url, size, session, min_request
        self.pos = 0
        self._buf_start = 0
        self._buf = b""
        self.bytes_fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def fetch(self, start: int, end_inclusive: int) -> bytes:
        r = self.session.get(
            self.url, headers={"Range": f"bytes={start}-{end_inclusive}"}, timeout=120
        )
        if r.status_code != 206:
            raise RuntimeError(
                f"expected 206 for range {start}-{end_inclusive}, got {r.status_code}"
            )
        self.bytes_fetched += len(r.content)
        return r.content

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self.pos
        end = min(self.pos + n, self.size)
        if not (self._buf_start <= self.pos and end <= self._buf_start + len(self._buf)):
            want = max(n, self.min_request)
            start = max(0, min(self.pos, self.size - want))
            self._buf = self.fetch(start, min(start + want, self.size) - 1)
            self._buf_start = start
        off = self.pos - self._buf_start
        out = self._buf[off : off + (end - self.pos)]
        self.pos += len(out)
        return out


def local_header_data_start(rf: HttpRangeFile, info: zipfile.ZipInfo) -> int:
    """Offset of the member's compressed bytes: local header (30 B) + its name and extra fields."""
    hdr = rf.fetch(info.header_offset, info.header_offset + 29)
    sig, _, _, _, _, _, _, _, _, name_len, extra_len = struct.unpack("<IHHHHHIIIHH", hdr)
    if sig != 0x04034B50:
        raise RuntimeError(f"{info.filename}: bad local header signature {sig:#x}")
    return info.header_offset + 30 + name_len + extra_len


def _decode(info: zipfile.ZipInfo, raw: bytes) -> bytes:
    if info.compress_type == zipfile.ZIP_STORED:
        data = raw
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(raw, -15)
    else:
        raise RuntimeError(f"{info.filename}: unsupported compression {info.compress_type}")
    if len(data) != info.file_size or (zlib.crc32(data) & 0xFFFFFFFF) != info.CRC:
        raise RuntimeError(f"{info.filename}: size/CRC mismatch after download")
    return data


def _write(info: zipfile.ZipInfo, data: bytes, out_root: Path) -> Path:
    out = out_root / PurePosixPath(info.filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return out


def extract_member(rf: HttpRangeFile, info: zipfile.ZipInfo, out_root: Path) -> Path:
    """One member = two small requests (local header, then data). Fine for big files."""
    start = local_header_data_start(rf, info)
    raw = rf.fetch(start, start + info.compress_size - 1)
    return _write(info, _decode(info, raw), out_root)


def extract_span(rf: HttpRangeFile, infos: list[zipfile.ZipInfo], out_root: Path) -> int:
    """Many small members that sit next to each other = one request for the whole byte span.

    Members are stored one after another (local header + data). We fetch from the first local header to
    the end of the last member's data in one streamed request and slice each member out locally. The
    local header length is read from the bytes themselves, so no per-member request is needed.
    """
    infos = sorted(infos, key=lambda i: i.header_offset)
    span_start = infos[0].header_offset
    last = infos[-1]
    # the last member's data end is not known until its local header is read; over-fetch by 64 KB
    span_end = min(TOTAL_SIZE, last.header_offset + 30 + 65536 + last.compress_size) - 1
    r = rf.session.get(
        rf.url, headers={"Range": f"bytes={span_start}-{span_end}"}, stream=True, timeout=600
    )
    if r.status_code != 206:
        raise RuntimeError(f"expected 206 for span, got {r.status_code}")
    buf = bytearray()
    for chunk in r.iter_content(8 << 20):
        buf += chunk
    rf.bytes_fetched += len(buf)
    n = 0
    for info in infos:
        off = info.header_offset - span_start
        sig, _, _, _, _, _, _, _, _, name_len, extra_len = struct.unpack(
            "<IHHHHHIIIHH", buf[off : off + 30]
        )
        if sig != 0x04034B50:
            raise RuntimeError(f"{info.filename}: bad local header signature {sig:#x}")
        data_off = off + 30 + name_len + extra_len
        _write(info, _decode(info, bytes(buf[data_off : data_off + info.compress_size])), out_root)
        n += 1
    return n


def group_spans(infos: list[zipfile.ZipInfo], max_gap: int) -> list[list[zipfile.ZipInfo]]:
    """Split members (sorted by offset) into runs whose neighbours are at most max_gap bytes apart."""
    infos = sorted(infos, key=lambda i: i.header_offset)
    spans: list[list[zipfile.ZipInfo]] = []
    for info in infos:
        if spans:
            prev = spans[-1][-1]
            prev_end = prev.header_offset + 30 + len(prev.filename.encode()) + prev.compress_size
            if info.header_offset - prev_end <= max_gap:
                spans[-1].append(info)
                continue
        spans.append([info])
    return spans


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--list", action="store_true", help="print the archive layout and exit")
    ap.add_argument("--include", help="regex on the member path; only matching files are fetched")
    ap.add_argument("--out-dir", type=Path, help="where members are written (archive paths kept)")
    ap.add_argument(
        "--dry-run", action="store_true", help="show what --include would fetch, download nothing"
    )
    ap.add_argument(
        "--bulk",
        action="store_true",
        help="fetch adjacent small members in one request per contiguous span (many small files)",
    )
    ap.add_argument(
        "--bulk-gap", type=int, default=1 << 20, help="max gap in bytes between members of one span"
    )
    args = ap.parse_args()
    if not args.list and not args.include:
        ap.error("give --list or --include")
    if args.include and not args.dry_run and not args.out_dir:
        ap.error("--include needs --out-dir (or --dry-run)")

    session = requests.Session()
    rf = HttpRangeFile(URL, TOTAL_SIZE, session)
    zf = zipfile.ZipFile(rf)  # reads only the central directory
    infos = [i for i in zf.infolist() if not i.is_dir()]
    print(
        f"[zip] {len(infos):,} files in the archive; directory read cost {rf.bytes_fetched / 1e6:.1f} MB"
    )

    by_dir = Counter()
    size_by_dir = Counter()
    for i in infos:
        d = "/".join(PurePosixPath(i.filename).parts[:3])
        by_dir[d] += 1
        size_by_dir[d] += i.file_size
    print("| folder (3 levels) | files | size GB |\n|---|---:|---:|")
    for d, n in sorted(by_dir.items()):
        print(f"| {d} | {n:,} | {size_by_dir[d] / 1e9:.2f} |")
    if args.list:
        return

    pat = re.compile(args.include)
    chosen = [i for i in infos if pat.search(i.filename)]
    total = sum(i.compress_size for i in chosen)
    print(
        f"[zip] --include {args.include!r}: {len(chosen):,} files, {total / 1e9:.2f} GB compressed"
    )
    for i in chosen[:5]:
        print(
            "   ",
            i.filename,
            f"{i.file_size / 1e6:.1f} MB",
            "deflate" if i.compress_type else "stored",
        )
    if args.dry_run:
        return

    if args.bulk:
        spans = group_spans(chosen, max_gap=args.bulk_gap)
        print(f"[zip] bulk mode: {len(spans)} contiguous span(s)")
        n = 0
        for span in tqdm(spans, unit="span"):
            n += extract_span(rf, span, args.out_dir)
    else:
        n = 0
        for i in tqdm(chosen, unit="file"):
            extract_member(rf, i, args.out_dir)
            n += 1
    print(
        f"[zip] wrote {n} files under {args.out_dir}; total fetched {rf.bytes_fetched / 1e9:.2f} GB"
    )


if __name__ == "__main__":
    sys.exit(main())
