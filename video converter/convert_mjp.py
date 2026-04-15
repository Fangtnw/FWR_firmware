#!/usr/bin/env python3
"""
Convert FWR_VISION .MJP JPEG-stream recordings to MP4.

File format:
  - 512-byte header
  - repeated records:
      uint32 little-endian jpeg_size
      jpeg_size bytes of JPEG payload

Optional sidecar:
  - matching .CSV for timestamp-derived FPS
"""

from __future__ import annotations

import argparse
import csv
import os
import struct
import sys

import cv2
import numpy as np

HEADER_FMT = "<8s6I120I"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def load_header(path: str) -> dict:
    with open(path, "rb") as f:
        raw = f.read(HEADER_SIZE)
    if len(raw) != HEADER_SIZE:
        raise ValueError("file too small for MJP header")
    magic, version, width, height, nominal_fps, quality, frame_count, *_ = struct.unpack(HEADER_FMT, raw)
    magic = magic.rstrip(b"\0")
    if magic != b"MJPGSEQ":
        raise ValueError(f"unexpected magic {magic!r}")
    return {
        "version": version,
        "width": width,
        "height": height,
        "nominal_fps": nominal_fps,
        "quality": quality,
        "frame_count": frame_count,
    }


def fps_from_csv(csv_path: str) -> float | None:
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 2:
        return None
    first = float(rows[0]["timestamp_ms"])
    last = float(rows[-1]["timestamp_ms"])
    if last <= first:
        return None
    return (len(rows) - 1) * 1000.0 / (last - first)


def iter_frames(path: str):
    with open(path, "rb") as f:
        f.seek(HEADER_SIZE)
        while True:
            len_raw = f.read(4)
            if not len_raw:
                break
            if len(len_raw) != 4:
                raise ValueError("truncated frame length")
            (frame_len,) = struct.unpack("<I", len_raw)
            payload = f.read(frame_len)
            if len(payload) != frame_len:
                raise ValueError("truncated jpeg payload")
            yield payload


def convert_file(path: str, output: str | None = None) -> str:
    header = load_header(path)
    csv_path = os.path.splitext(path)[0] + ".CSV"
    fps = fps_from_csv(csv_path) or max(1.0, float(header["nominal_fps"] or 30))

    if output is None:
        output = os.path.splitext(path)[0] + ".mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (header["width"], header["height"]))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open output writer: {output}")

    frame_idx = 0
    try:
        for payload in iter_frames(path):
            arr = np.frombuffer(payload, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"failed to decode JPEG at frame {frame_idx}")
            if frame.shape[1] != header["width"] or frame.shape[0] != header["height"]:
                frame = cv2.resize(frame, (header["width"], header["height"]), interpolation=cv2.INTER_LINEAR)
            writer.write(frame)
            frame_idx += 1
    finally:
        writer.release()

    print(f"Converted {frame_idx} frames at {fps:.2f} fps -> {output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert FWR_VISION .MJP JPEG-stream recordings to MP4")
    parser.add_argument("input", help=".MJP input file")
    parser.add_argument("-o", "--output", help="output .mp4 path")
    args = parser.parse_args()

    try:
        convert_file(args.input, args.output)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
