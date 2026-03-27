#!/usr/bin/env python3
"""
FWR_VISION .VID → MP4 converter
================================
Reads RAW RGB565 video files recorded by the ESP32-P4 firmware and converts
them to standard MP4.

File format
-----------
  Bytes 0–511  : 512-byte header sector
    offset  0  : uint32  frame_count
    offset  4  : uint32  fps
    offset  8  : uint32  width
    offset 12  : uint32  height
    offset 16  : uint32  frame_size  (bytes per frame = width * height * 2)
    offset 20+ : (padding, zeros)
  Bytes 512+   : frame_count × frame_size bytes, each frame packed RGB565

Optional sidecar
----------------
  <same-name>.CSV  — OFD (optical flow divergence) data, one row per frame.
  Overlay is shown when --ofd flag is passed.

Requirements
------------
  pip install opencv-python numpy

Usage
-----
  python convert_vid.py V0000.VID                        # → V0000.mp4
  python convert_vid.py V0000.VID recording.mp4          # custom output
  python convert_vid.py V0000.VID --ofd                  # OFD overlay
  python convert_vid.py V0000.VID recording.mp4 --ofd
  python convert_vid.py *.VID                            # batch convert
"""

import struct
import sys
import os
import argparse
import csv as csv_mod
from glob import glob

import numpy as np
try:
    import cv2
except ImportError:
    sys.exit("opencv-python not installed.\n  Run: pip install opencv-python numpy")

# ── constants ──────────────────────────────────────────────────────────────────
HEADER_SIZE = 512       # bytes (1 sector, sector-aligned)
HEADER_FMT  = "<5I"     # frame_count, fps, width, height, frame_size (all uint32 LE)


# ── RGB565 → BGR888 ────────────────────────────────────────────────────────────
def rgb565_to_bgr(data: bytes, width: int, height: int) -> np.ndarray:
    """Convert packed little-endian RGB565 bytes to a (H, W, 3) BGR uint8 array."""
    px = np.frombuffer(data, dtype="<u2")          # uint16, little-endian
    r = ((px >> 11) & 0x1F).astype(np.uint32) * 255 // 31
    g = ((px >>  5) & 0x3F).astype(np.uint32) * 255 // 63
    b = ( px        & 0x1F).astype(np.uint32) * 255 // 31
    bgr = np.stack([b, g, r], axis=1).astype(np.uint8).reshape(height, width, 3)
    return bgr


# ── OFD sidecar ────────────────────────────────────────────────────────────────
def load_ofd(csv_path: str) -> dict:
    """Return {frame_number: row_dict} from OFD CSV sidecar, or {} if not found."""
    rows = {}
    try:
        with open(csv_path, newline="") as f:
            for row in csv_mod.DictReader(f):
                rows[int(row["frame"])] = row
        print(f"  OFD: loaded {len(rows)} rows from {os.path.basename(csv_path)}")
    except FileNotFoundError:
        print(f"  OFD: sidecar not found ({os.path.basename(csv_path)})")
    return rows


def draw_ofd_overlay(frame: np.ndarray, row: dict) -> None:
    """Draw divergence / tau overlay onto a BGR frame in-place."""
    valid = int(row.get("valid", 0))
    div   = float(row.get("divergence", 0.0))
    tau   = float(row.get("tau", 0.0))
    h     = frame.shape[0]

    if valid:
        label = f"div={div:+.3f}  tau={tau:.2f}s"
        color = (0, 230, 0)
    else:
        label = "OFD: no valid flow"
        color = (0, 100, 220)

    cv2.putText(frame, label, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color,   1, cv2.LINE_AA)


# ── main conversion ────────────────────────────────────────────────────────────
def convert(vid_path: str, out_path: str = None, show_ofd: bool = False) -> bool:
    if out_path is None:
        out_path = os.path.splitext(vid_path)[0] + ".mp4"

    csv_path = os.path.splitext(vid_path)[0] + ".CSV"
    if not os.path.exists(csv_path):
        csv_path = os.path.splitext(vid_path)[0] + ".csv"

    print(f"\n{'─'*60}")
    print(f"Input  : {vid_path}")
    print(f"Output : {out_path}")

    # ── read header ────────────────────────────────────────────────────────────
    with open(vid_path, "rb") as f:
        hdr_bytes = f.read(HEADER_SIZE)

    if len(hdr_bytes) < HEADER_SIZE:
        print("ERROR: file too small to contain a valid header.")
        return False

    frame_count, fps, width, height, frame_size = struct.unpack_from(HEADER_FMT, hdr_bytes)

    # Sanity / fallback values
    if frame_size == 0:
        frame_size = width * height * 2
        print(f"  frame_size was 0 — assuming RGB565: {frame_size} B")
    if fps == 0:
        fps = 10
        print(f"  fps was 0 — defaulting to {fps}")

    print(f"Header : {frame_count} frames × {frame_size} B  |  "
          f"{width}×{height} px  |  {fps} fps")

    if frame_count == 0 or width == 0 or height == 0:
        print("ERROR: header contains invalid dimensions.")
        return False

    # ── optional OFD ───────────────────────────────────────────────────────────
    ofd = load_ofd(csv_path) if show_ofd else {}

    # ── set up writer ──────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (width, height))
    if not writer.isOpened():
        print(f"ERROR: could not open output file: {out_path}")
        return False

    # ── convert frames ─────────────────────────────────────────────────────────
    written = 0
    with open(vid_path, "rb") as f:
        f.seek(HEADER_SIZE)
        for i in range(frame_count):
            raw = f.read(frame_size)
            if len(raw) < frame_size:
                print(f"\n  WARNING: file truncated at frame {i + 1}")
                break

            bgr = rgb565_to_bgr(raw, width, height)

            if ofd and (i + 1) in ofd:
                draw_ofd_overlay(bgr, ofd[i + 1])

            writer.write(bgr)
            written += 1

            if written % 30 == 0 or written == frame_count:
                pct = written / frame_count * 100
                bar = "█" * (written * 30 // frame_count)
                bar = bar.ljust(30)
                print(f"  [{bar}] {written}/{frame_count} ({pct:.0f}%)", end="\r")

    writer.release()
    print(f"\nDone   : {written} frames written → {out_path}")
    return True


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Convert FWR_VISION .VID (RAW RGB565) files to MP4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("input", nargs="+",
                    help=".VID file(s) or glob pattern (e.g. '*.VID')")
    ap.add_argument("-o", "--output", default=None,
                    help="Output .mp4 path (only valid with a single input file)")
    ap.add_argument("--ofd", action="store_true",
                    help="Overlay OFD divergence / tau from sidecar .CSV")
    args = ap.parse_args()

    # Expand glob patterns (needed on Windows where shell doesn't expand them)
    inputs = []
    for pat in args.input:
        expanded = glob(pat)
        inputs.extend(expanded if expanded else [pat])

    if len(inputs) > 1 and args.output:
        ap.error("--output / -o can only be used with a single input file")

    ok = 0
    for vid in inputs:
        if not os.path.isfile(vid):
            print(f"SKIP: {vid} (file not found)")
            continue
        out = args.output if len(inputs) == 1 else None
        if convert(vid, out, show_ofd=args.ofd):
            ok += 1

    print(f"\n{'─'*60}")
    print(f"Converted {ok}/{len(inputs)} file(s).")


if __name__ == "__main__":
    main()
