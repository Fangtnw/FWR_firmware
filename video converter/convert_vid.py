#!/usr/bin/env python3
"""
FWR_VISION .VID → MP4 converter  (with OFD debug panel)
=========================================================
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
  <same-name>.CSV  — OFD + IMU data, one row per frame.

  Two CSV schemas are handled automatically:

  OLD (pre-optimisation):
    frame, timestamp_ms, divergence, lr_balance, tau, vx_mean, vy_mean,
    flow_cnt, div_cnt, valid,
    ax, ay, az, gx, gy, gz, roll, pitch, yaw

  NEW (post-optimisation — ofd_config.h era):
    frame, timestamp_ms, divergence, lr_balance, tau, vx_mean, vy_mean,
    flow_cnt, div_cnt, valid,
    ema_div, ema_lr, tau_ms, looming, evasion_level, turn_cmd, az_quiet,
    ax, ay, az, gx, gy, gz, roll, pitch, yaw

Requirements
------------
  pip install opencv-python numpy

Usage
-----
  python convert_vid.py V0000.VID                  → plain MP4
  python convert_vid.py V0000.VID --ofd            → MP4 + OFD debug panel
  python convert_vid.py V0000.VID --ofd --ofd-lite → MP4 + single-line overlay
  python convert_vid.py V0000.VID -o out.mp4 --ofd
  python convert_vid.py *.VID --ofd                → batch convert with panel
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
HEADER_SIZE = 512
HEADER_FMT  = "<5I"   # frame_count, fps, width, height, frame_size (uint32 LE)

# OFD constants (must match ofd_config.h) — used for bar scales
OFD_TAU_EVADE_MS  = 30.0
OFD_TAU_BRAKE_MS  = 50.0
OFD_TAU_MAX       = 1000.0
OFD_DIV_THRESHOLD = 0.05
OFD_AZ_QUIET_CENTER = -0.986
OFD_AZ_QUIET_BAND   =  0.150

# Side-panel dimensions (appended to the right of the camera frame)
PANEL_W = 260   # pixels wide
FS  = 0.38      # font scale for data values
FSL = 0.40      # font scale for labels/headers
FT  = 1         # font thickness (data)
LH  = 13        # line height (pixels)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Evasion level → label/color
EVADE_INFO = {
    0: ("  OK  ",  (40, 160, 40)),
    1: (" ALERT",  (0, 200, 200)),
    2: (" BRAKE",  (0, 140, 255)),
    3: (" EVADE",  (0, 40, 220)),
}
INVALID_INFO = ("INVALID", (100, 100, 100))
GATE_INFO    = ("  SKIP ",  (80, 80, 80))   # wing-sync gate blocked


# ── RGB565 → BGR888 ────────────────────────────────────────────────────────────
def rgb565_to_bgr(data: bytes, width: int, height: int) -> np.ndarray:
    """Convert packed little-endian RGB565 bytes to a (H, W, 3) BGR uint8 array."""
    px = np.frombuffer(data, dtype="<u2")
    r = ((px >> 11) & 0x1F).astype(np.uint32) * 255 // 31
    g = ((px >>  5) & 0x3F).astype(np.uint32) * 255 // 63
    b = ( px        & 0x1F).astype(np.uint32) * 255 // 31
    return np.stack([b, g, r], axis=1).astype(np.uint8).reshape(height, width, 3)


# ── OFD sidecar ────────────────────────────────────────────────────────────────
def load_ofd(csv_path: str) -> tuple[dict, bool]:
    """
    Return ({frame_number: row_dict}, has_new_columns) from OFD CSV sidecar,
    or ({}, False) if file not found.
    """
    rows = {}
    has_new = False
    try:
        with open(csv_path, newline="") as f:
            reader = csv_mod.DictReader(f)
            fieldnames = reader.fieldnames or []
            has_new = "ema_div" in fieldnames
            for row in reader:
                rows[int(row["frame"])] = row
        schema = "new (ema_div + filter columns)" if has_new else "old (raw only)"
        print(f"  OFD: loaded {len(rows)} rows  |  schema: {schema}")
    except FileNotFoundError:
        print(f"  OFD: sidecar not found ({os.path.basename(csv_path)})")
    return rows, has_new


# ── drawing helpers ─────────────────────────────────────────────────────────────
def _txt(img, text, x, y, color=(220, 220, 220), scale=FS, thickness=FT):
    """Draw shadowed text for readability on any background."""
    cv2.putText(img, text, (x + 1, y + 1), FONT, scale,
                (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def _section(img, label, x, y, w):
    """Draw a horizontal section divider with label."""
    mid = y - 4
    cv2.line(img, (x, mid), (x + w - 4, mid), (70, 70, 70), 1)
    _txt(img, label, x + 2, y, color=(150, 150, 150), scale=FS - 0.02)


def _hbar(img, x, y, w, h_bar, value, vmin, vmax,
          col_lo=(50, 50, 200), col_hi=(50, 200, 50), col_bg=(40, 40, 40)):
    """
    Horizontal bar: value in [vmin, vmax] drawn from left.
    Returns the bar as a filled rectangle.
    """
    cv2.rectangle(img, (x, y), (x + w, y + h_bar), col_bg, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h_bar), (80, 80, 80), 1)
    frac = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    fill_w = int(frac * w)
    if fill_w > 0:
        t = frac  # interpolate color
        col = tuple(int(col_lo[i] + t * (col_hi[i] - col_lo[i])) for i in range(3))
        cv2.rectangle(img, (x, y + 1), (x + fill_w, y + h_bar - 1), col, -1)


def _cbar(img, x, y, w, h_bar, value, vmin, vmax,
          col_neg=(50, 50, 200), col_pos=(50, 200, 50), col_bg=(40, 40, 40)):
    """
    Centered bar: value=0 is in the middle; negative goes left, positive goes right.
    """
    cv2.rectangle(img, (x, y), (x + w, y + h_bar), col_bg, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h_bar), (80, 80, 80), 1)
    cx = x + w // 2
    cv2.line(img, (cx, y), (cx, y + h_bar), (120, 120, 120), 1)
    frac = max(-1.0, min(1.0, value / max(abs(vmin), abs(vmax))))
    fill_w = int(abs(frac) * (w // 2))
    if fill_w > 0:
        if frac >= 0:
            cv2.rectangle(img, (cx, y + 1), (cx + fill_w, y + h_bar - 1), col_pos, -1)
        else:
            cv2.rectangle(img, (cx - fill_w, y + 1), (cx, y + h_bar - 1), col_neg, -1)


def _tau_bar(img, x, y, w, h_bar, tau_ms):
    """
    τ bar: full width = OFD_TAU_MAX (1000ms). Color transitions red→orange→green
    as τ increases. A threshold line is drawn at BRAKE and EVADE.
    """
    # background
    cv2.rectangle(img, (x, y), (x + w, y + h_bar), (40, 40, 40), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h_bar), (80, 80, 80), 1)

    frac = min(1.0, tau_ms / OFD_TAU_MAX)
    fill_w = max(1, int(frac * w))

    # gradient: red at 0ms, orange at 50ms, green at 200ms+
    if tau_ms < OFD_TAU_EVADE_MS:
        col = (0, 30, 220)     # red
    elif tau_ms < OFD_TAU_BRAKE_MS:
        col = (0, 130, 255)    # orange
    elif tau_ms < 100:
        col = (0, 200, 200)    # yellow
    else:
        col = (50, 200, 50)    # green

    cv2.rectangle(img, (x, y + 1), (x + fill_w, y + h_bar - 1), col, -1)

    # threshold markers
    for thr_ms, thr_col in [(OFD_TAU_EVADE_MS, (0, 0, 255)),
                             (OFD_TAU_BRAKE_MS, (0, 80, 255))]:
        mx = x + int(thr_ms / OFD_TAU_MAX * w)
        cv2.line(img, (mx, y), (mx, y + h_bar), thr_col, 1)


def _status_badge(panel, x, y, w, h, label, color):
    """Draw a large colored status badge with centered label."""
    cv2.rectangle(panel, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), (200, 200, 200), 1)
    scale = 0.7
    (tw, th), _ = cv2.getTextSize(label, FONT, scale, 2)
    tx = x + (w - tw) // 2
    ty = y + (h + th) // 2
    cv2.putText(panel, label, (tx + 1, ty + 1), FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(panel, label, (tx, ty), FONT, scale, (240, 240, 240), 2, cv2.LINE_AA)


# ── panel builder ──────────────────────────────────────────────────────────────
def build_ofd_panel(row: dict, has_new: bool, frame_h: int) -> np.ndarray:
    """
    Build a PANEL_W × frame_h debug panel for one CSV row.
    Returns a (frame_h, PANEL_W, 3) BGR uint8 array.
    """
    panel = np.full((frame_h, PANEL_W, 3), (18, 18, 18), dtype=np.uint8)
    pw = PANEL_W
    y = 0

    # ── title bar ─────────────────────────────────────────────────────────────
    cv2.rectangle(panel, (0, 0), (pw, 16), (35, 35, 60), -1)
    ts  = float(row.get("timestamp_ms", 0)) / 1000.0
    frm = int(row.get("frame", 0))
    _txt(panel, f"OFD  F:{frm:04d}  t={ts:.2f}s",
         3, 12, color=(180, 180, 255), scale=FS)
    y = 18

    # ── parse common fields ───────────────────────────────────────────────────
    valid    = int(row.get("valid",    0))
    div      = float(row.get("divergence", 0.0))
    tau_raw  = float(row.get("tau",   0.0))
    lr       = float(row.get("lr_balance", 0.0))
    vx       = float(row.get("vx_mean", 0.0))
    vy       = float(row.get("vy_mean", 0.0))
    flow_cnt = int(row.get("flow_cnt", 0))
    div_cnt  = int(row.get("div_cnt",  0))
    ax = float(row.get("ax", 0.0)); ay = float(row.get("ay", 0.0))
    az = float(row.get("az", 0.0))
    gx = float(row.get("gx", 0.0)); gy = float(row.get("gy", 0.0))
    gz = float(row.get("gz", 0.0))
    roll  = float(row.get("roll",  0.0))
    pitch = float(row.get("pitch", 0.0))
    yaw   = float(row.get("yaw",   0.0))

    # ── parse new filter fields (may be absent in old CSVs) ──────────────────
    if has_new:
        ema_div     = float(row.get("ema_div",      0.0))
        ema_lr      = float(row.get("ema_lr",       0.0))
        tau_ms      = float(row.get("tau_ms",       OFD_TAU_MAX))
        looming     = int(row.get("looming",        0))
        evade_lvl   = int(row.get("evasion_level",  0))
        turn_cmd    = float(row.get("turn_cmd",     0.0))
        az_quiet    = int(row.get("az_quiet",       0))
    else:
        ema_div = div - (-0.018)      # approximate bias-corrected value for old CSVs
        ema_lr  = lr
        tau_ms  = OFD_TAU_MAX
        looming = 0
        evade_lvl = 0
        turn_cmd  = 0.0
        az_quiet  = -1    # unknown for old CSV

    # ── status badge ─────────────────────────────────────────────────────────
    if not valid:
        badge_label, badge_col = INVALID_INFO
    elif has_new and az_quiet == 0:
        badge_label, badge_col = GATE_INFO
    elif has_new:
        info = EVADE_INFO.get(evade_lvl, EVADE_INFO[0])
        badge_label, badge_col = info
    else:
        # old CSV: derive status from raw divergence threshold
        if valid and div > OFD_DIV_THRESHOLD:
            badge_label, badge_col = EVADE_INFO[2]   # show as BRAKE if above threshold
        else:
            badge_label, badge_col = EVADE_INFO[0]

    _status_badge(panel, 4, y, pw - 8, 28, badge_label, badge_col)
    y += 32

    # ── τ bar (with threshold markers) ───────────────────────────────────────
    t_disp = tau_ms if has_new else (1.0 / div * 1000.0 if div > 1e-6 else OFD_TAU_MAX)
    t_disp = min(t_disp, OFD_TAU_MAX)
    _txt(panel, f"tau_ms: {t_disp:6.1f}  (evd={OFD_TAU_EVADE_MS:.0f} brk={OFD_TAU_BRAKE_MS:.0f})",
         4, y + 10, color=(210, 210, 210), scale=FS - 0.02)
    y += 13
    _tau_bar(panel, 4, y, pw - 8, 9, t_disp)
    y += 13

    # ── Detection section ────────────────────────────────────────────────────
    y += 2
    _section(panel, "Detection / Filter", 4, y, pw - 4)
    y += 6

    if has_new:
        loom_str  = "YES" if looming else " no"
        loom_col  = (50, 255, 50) if looming else (160, 160, 160)
        lvl_names = {0: "NONE", 1: "ALERT", 2: "BRAKE", 3: "EVADE"}
        _txt(panel, f"looming: {loom_str}    lvl: {lvl_names.get(evade_lvl, '?')}",
             4, y + 10, color=loom_col)
        y += LH

    # ema_div bar
    _txt(panel, f"ema_div: {ema_div:+.4f}  thr=\xb1{OFD_DIV_THRESHOLD:.3f}",
         4, y + 10, color=(210, 210, 210), scale=FS - 0.02)
    y += 13
    _cbar(panel, 4, y, pw - 8, 8, ema_div, -0.15, 0.15,
          col_neg=(50, 50, 180), col_pos=(50, 180, 50))
    y += 11

    # ema_lr / turn_cmd
    if has_new:
        _txt(panel, f"ema_lr: {ema_lr:+.4f}   turn: {turn_cmd:+.3f}",
             4, y + 10, color=(210, 210, 210))
        y += LH
        _txt(panel, f"turn_cmd:", 4, y + 10, color=(160, 160, 160), scale=FS - 0.02)
        _cbar(panel, 65, y + 3, pw - 69, 8, turn_cmd, -1.0, 1.0,
              col_neg=(180, 50, 50), col_pos=(50, 50, 180))
        y += 13
    else:
        _txt(panel, f"lr:     {lr:+.4f}",
             4, y + 10, color=(210, 210, 210))
        y += LH

    # ── Raw OFD section ──────────────────────────────────────────────────────
    y += 2
    _section(panel, "Raw OFD", 4, y, pw - 4)
    y += 6

    div_col = (50, 220, 50) if div > OFD_DIV_THRESHOLD else (180, 180, 180)
    _txt(panel, f"div:  {div:+.5f}   tau: {tau_raw:.4f}s",
         4, y + 10, color=div_col)
    y += LH
    _txt(panel, f"vx: {vx:+.2f}   vy: {vy:+.2f}  (flipped)",
         4, y + 10, color=(180, 180, 180), scale=FS - 0.02)
    y += LH
    valid_col = (50, 220, 50) if valid else (80, 80, 200)
    _txt(panel, f"flow:{flow_cnt:4d}  div_cnt:{div_cnt:4d}  v:{valid}",
         4, y + 10, color=valid_col, scale=FS - 0.02)
    y += LH

    # wing-sync gate
    if has_new:
        az_in_band = abs(az - OFD_AZ_QUIET_CENTER) < OFD_AZ_QUIET_BAND
        gate_col = (50, 220, 50) if az_quiet else (60, 60, 180)
        gate_str = "OPEN" if az_quiet else "BLOCKED"
        _txt(panel, f"az_quiet: {gate_str}",
             4, y + 10, color=gate_col)
        y += LH

    # ── IMU section ──────────────────────────────────────────────────────────
    y += 2
    _section(panel, "IMU", 4, y, pw - 4)
    y += 6

    roll_col = (50, 200, 200) if abs(roll + 180) < 30 else (80, 80, 200)
    _txt(panel, f"roll:{roll:7.1f}\xb0  pitch:{pitch:5.1f}\xb0",
         4, y + 10, color=roll_col)
    y += LH
    _txt(panel, f"yaw: {yaw:7.1f}\xb0",
         4, y + 10, color=(180, 180, 180))
    y += LH

    # az with wing-sync band indicator
    az_band_lo = OFD_AZ_QUIET_CENTER - OFD_AZ_QUIET_BAND
    az_band_hi = OFD_AZ_QUIET_CENTER + OFD_AZ_QUIET_BAND
    az_col = (50, 220, 50) if az_band_lo < az < az_band_hi else (80, 80, 200)
    _txt(panel, f"az: {az:+.3f}g  [{az_band_lo:.2f},{az_band_hi:.2f}]",
         4, y + 10, color=az_col)
    y += LH
    # az mini bar (centered at OFD_AZ_QUIET_CENTER)
    _txt(panel, "az:", 4, y + 9, color=(120, 120, 120), scale=FS - 0.04)
    _cbar(panel, 28, y + 2, pw - 32, 8,
          az - OFD_AZ_QUIET_CENTER, -0.5, 0.5,
          col_neg=(80, 80, 180), col_pos=(80, 80, 180))
    # draw band limits
    cx_az = 28 + (pw - 32) // 2
    bw_az = int(OFD_AZ_QUIET_BAND / 0.5 * ((pw - 32) // 2))
    cv2.rectangle(panel, (cx_az - bw_az, y + 2), (cx_az + bw_az, y + 10),
                  (50, 180, 50), 1)
    y += 13

    _txt(panel, f"ax:{ax:+.3f}g  ay:{ay:+.3f}g",
         4, y + 10, color=(160, 160, 160), scale=FS - 0.03)
    y += LH
    _txt(panel, f"gx:{gx:+.1f}  gy:{gy:+.1f}  gz:{gz:+.1f} dps",
         4, y + 10, color=(140, 140, 140), scale=FS - 0.04)

    # ── bottom hint ──────────────────────────────────────────────────────────
    hint = "new CSV" if has_new else "old CSV (no filter cols)"
    cv2.rectangle(panel, (0, frame_h - 14), (pw, frame_h), (30, 30, 30), -1)
    _txt(panel, hint, 4, frame_h - 4, color=(100, 100, 100), scale=FS - 0.06)

    return panel


# ── lite overlay (single-line, backward-compat) ────────────────────────────────
def draw_ofd_overlay_lite(frame: np.ndarray, row: dict, has_new: bool) -> None:
    """Draw a minimal one-line overlay onto frame in-place (original behaviour)."""
    valid   = int(row.get("valid", 0))
    div     = float(row.get("divergence", 0.0))
    h       = frame.shape[0]

    if has_new:
        tau_ms    = float(row.get("tau_ms", OFD_TAU_MAX))
        looming   = int(row.get("looming", 0))
        ema_div   = float(row.get("ema_div", 0.0))
        az_quiet  = int(row.get("az_quiet", 1))
        evade_lvl = int(row.get("evasion_level", 0))
        lvl_names = {0: "OK", 1: "ALERT", 2: "BRAKE", 3: "EVADE"}
        label = (f"[{lvl_names[evade_lvl]}] "
                 f"ema={ema_div:+.3f}  tau={tau_ms:.1f}ms  "
                 f"gate={'ON' if az_quiet else 'OFF'}")
        color = EVADE_INFO.get(evade_lvl, EVADE_INFO[0])[1] if valid else (100, 100, 100)
    elif valid:
        tau = float(row.get("tau", 0.0))
        label = f"div={div:+.3f}  tau={tau:.2f}s"
        color = (0, 230, 0)
    else:
        label = "OFD: no valid flow"
        color = (0, 100, 220)

    cv2.putText(frame, label, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)


# ── main conversion ────────────────────────────────────────────────────────────
def convert(vid_path: str, out_path: str = None,
            show_ofd: bool = False, ofd_lite: bool = False) -> bool:

    if out_path is None:
        stem = os.path.splitext(vid_path)[0]
        if show_ofd and ofd_lite:
            suffix = "_ofd_lite"
        elif show_ofd:
            suffix = "_ofd_panel"
        else:
            suffix = ""
        out_path = stem + suffix + ".mp4"

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

    # ── load OFD sidecar ───────────────────────────────────────────────────────
    ofd, has_new = load_ofd(csv_path) if show_ofd else ({}, False)

    # ── set up video writer ────────────────────────────────────────────────────
    # Output width is wider when side-panel is active.
    # +2 accounts for the 2-px separator column between the camera frame and the panel.
    SEP_W = 2
    use_panel = show_ofd and not ofd_lite
    out_w = width + SEP_W + PANEL_W if use_panel else width
    out_h = height

    if use_panel:
        print(f"Mode   : OFD debug panel  ({width}×{height} + {SEP_W}px sep + {PANEL_W}px panel → {out_w}×{out_h})")
    elif show_ofd:
        print(f"Mode   : OFD lite overlay  ({width}×{height})")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (out_w, out_h))
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
            frame_num = i + 1
            row = ofd.get(frame_num)

            if row is not None:
                if use_panel:
                    panel = build_ofd_panel(row, has_new, height)
                    # draw vertical separator
                    sep = np.full((height, 2, 3), (60, 60, 60), dtype=np.uint8)
                    bgr = np.concatenate([bgr, sep, panel], axis=1)
                else:
                    draw_ofd_overlay_lite(bgr, row, has_new)
            elif use_panel and show_ofd:
                # no CSV row for this frame — blank panel
                blank = np.full((height, PANEL_W + 2, 3), (18, 18, 18), dtype=np.uint8)
                cv2.putText(blank, f"F:{frame_num:04d} no data",
                            (4, 18), FONT, FS, (80, 80, 80), 1, cv2.LINE_AA)
                bgr = np.concatenate([bgr, blank], axis=1)

            writer.write(bgr)
            written += 1

            if written % 30 == 0 or written == frame_count:
                pct = written / frame_count * 100
                bar = ("█" * (written * 30 // frame_count)).ljust(30)
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
                    help="Append OFD+IMU debug panel to the right of each frame "
                         "(requires matching .CSV sidecar)")
    ap.add_argument("--ofd-lite", action="store_true",
                    help="Single-line OFD overlay instead of the full debug panel")
    args = ap.parse_args()

    inputs = []
    for pat in args.input:
        expanded = glob(pat)
        inputs.extend(expanded if expanded else [pat])

    if len(inputs) > 1 and args.output:
        ap.error("--output / -o can only be used with a single input file")

    show_ofd = args.ofd or args.ofd_lite

    ok = 0
    for vid in inputs:
        if not os.path.isfile(vid):
            print(f"SKIP: {vid} (file not found)")
            continue
        out = args.output if len(inputs) == 1 else None
        if convert(vid, out, show_ofd=show_ofd, ofd_lite=args.ofd_lite):
            ok += 1

    print(f"\n{'─'*60}")
    print(f"Converted {ok}/{len(inputs)} file(s).")


if __name__ == "__main__":
    main()
