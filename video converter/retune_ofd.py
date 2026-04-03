#!/usr/bin/env python3
"""
retune_ofd.py  —  OFD Full Re-processing & Parameter Tuning
=============================================================
Reads a .VID + optional .CSV (IMU sidecar), re-runs the complete OFD pipeline
from raw video frames with user-specified parameters, and exports an MP4 with
the same debug panel as convert_vid.py.

The raw OFD block-matching is ported 1-to-1 from ofd.cpp, so all tunables
(GRID_STEP, BLK_R, SRCH_R, MIN_TEX, MAX_SAD, EPS_DIV) and the OFD processing
resolution can be adjusted here and their effect visualised immediately.

CSV is used for IMU display and the az wing-sync gate only.
If no CSV is found, IMU fields are zero and the az gate is always open.

Usage
-----
  # All defaults — matches exact firmware behaviour:
  python retune_ofd.py V0008.VID

  # Tune filter layer only:
  python retune_ofd.py V0008.VID --alpha 0.5 --tau-evade 25

  # Tune block-matching:
  python retune_ofd.py V0008.VID --grid-step 8 --min-tex 15 --max-sad 6000

  # Run OFD at full recording resolution instead of default 160×120:
  python retune_ofd.py V0008.VID --ofd-res 320x240

Notes
-----
  - The tau-bar threshold markers are cosmetic and fixed at 30ms/50ms from
    convert_vid.py.  The status badge and bar fill correctly reflect your
    --tau-evade / --tau-brake values.
  - Orange left-stripe = evasion_level or looming changed vs original CSV.
    Only shown when the input CSV contains original filter columns.
"""

import struct
import sys
import os
import argparse
import csv as csv_mod
import importlib.util
import pathlib

import numpy as np
try:
    import cv2
except ImportError:
    sys.exit("opencv-python not installed.\n  Run: pip install opencv-python numpy")

# ── sliding_window_view (numpy ≥1.20); fall back gracefully ──────────────────
try:
    from numpy.lib.stride_tricks import sliding_window_view as _swv
    _HAVE_SWV = True
except ImportError:
    _HAVE_SWV = False

# ── import convert_vid utilities ───────────────────────────────────────────────
_HERE = pathlib.Path(__file__).parent
_cv_path = _HERE / "convert_vid.py"
if not _cv_path.exists():
    sys.exit(f"ERROR: convert_vid.py not found at {_cv_path}")
_spec = importlib.util.spec_from_file_location("convert_vid", _cv_path)
_cv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cv)


# ══════════════════════════════════════════════════════════════════════════════
#  OFD TUNING PARAMETERS
#  Edit these directly in this file for quick iteration, OR override per-run
#  with CLI flags shown in the argparse section below.
# ══════════════════════════════════════════════════════════════════════════════

# ── Filter layer ──────────────────────────────────────────────────────────────
OFD_DIV_BIAS        = -0.018   # DC offset subtracted from raw divergence
OFD_EMA_ALPHA       =  0.30    # EMA smoothing factor (0=frozen, 1=no memory)
OFD_MIN_DIV_CNT     = 10       # min valid divergence measurements per frame
OFD_DIV_THRESHOLD   =  0.05    # looming gate: |ema_div| must exceed this
OFD_TAU_EVADE_MS    = 30.0     # immediate evasion threshold (ms)
OFD_TAU_BRAKE_MS    = 50.0     # dual-gate outer / brake threshold (ms)
OFD_DT_MIN_MS       =  8.0     # dt clamp minimum (ms)
OFD_DT_MAX_MS       = 80.0     # dt clamp maximum (ms)
OFD_AZ_QUIET_CENTER = -0.986   # gravity-only z-accel reference (g)
OFD_AZ_QUIET_BAND   =  0.15    # ±band around quiet center (g)
OFD_LR_GAIN         =  3.0     # lr_balance → turn_cmd gain
OFD_TAU_MAX         = 1000.0   # sentinel value: no obstacle (ms)

# ── Block-matching (mirrors ofd.cpp tunables) ─────────────────────────────────
OFD_GRID_STEP  = 12      # spacing between flow-point grid (pixels, in OFD frame)
OFD_BLK_R      =  4      # block radius  →  block size = (2R+1)² = 9×9
OFD_SRCH_R     =  6      # search radius in previous frame (pixels)
OFD_MIN_TEX    = 25      # minimum texture score to accept a grid point
OFD_MAX_SAD    = 4500    # reject block match if SAD exceeds this
OFD_EPS_DIV    = 1e-6    # epsilon for tau = 1/div guard

# ── OFD processing resolution ─────────────────────────────────────────────────
# Format: "WxH".  Default = half of recording res (160×120 for 320×240 source).
# Examples: "80x60"  "160x120"  "240x180"  "320x240"
OFD_RESOLUTION = "320x240"

# ══════════════════════════════════════════════════════════════════════════════


# ── helpers ────────────────────────────────────────────────────────────────────
def _parse_res(res_str: str):
    """Parse "WxH" string → (w, h) int tuple."""
    try:
        w, h = res_str.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError(
            f"Invalid resolution '{res_str}' — expected WxH e.g. 160x120")


def _empty_ofd():
    return dict(divergence=0.0, tau=0.0, vx_mean=0.0, vy_mean=0.0,
                lr_balance=0.0, flow_cnt=0, div_cnt=0, valid=0)


# ── block-matching (1-to-1 port of ofd.cpp:ofd_process_gray) ──────────────────
def ofd_process_gray(cur: np.ndarray, prev: np.ndarray, params: dict) -> dict:
    """
    Python port of ofd_process_gray() from ofd.cpp.
    cur, prev: 2D uint8 grayscale arrays of same shape (H, W).
    Returns dict with raw OFD fields.  valid=0 when insufficient data.
    """
    gH, gW = cur.shape
    GRID_STEP   = params["grid_step"]
    BLK_R       = params["blk_r"]
    SRCH_R      = params["srch_r"]
    MIN_TEX     = params["min_tex"]
    MAX_SAD     = params["max_sad"]
    EPS_DIV     = params["eps_div"]
    MIN_DIV_CNT = params["min_div_cnt"]

    margin = BLK_R + SRCH_R + 2
    x0 = margin;  x1 = gW - margin - 1
    y0 = margin;  y1 = gH - margin - 1

    if x1 <= x0 or y1 <= y0:
        return _empty_ofd()

    blk_sz = 2 * BLK_R + 1
    srch_n = 2 * SRCH_R + 1

    # ── precompute texture image ────────────────────────────────────────────
    # tex[y, x] = |img[y, x+1] - img[y, x-1]| + |img[y+1, x] - img[y-1, x]|
    # matches: gx = p[i+1] - p[i-1],  gy = p[i+w] - p[i-w]  from ofd.cpp
    cur_i = cur.astype(np.int32)
    grad_x = np.zeros((gH, gW), np.int32)
    grad_y = np.zeros((gH, gW), np.int32)
    grad_x[:, 1:-1] = np.abs(cur_i[:, 2:] - cur_i[:, :-2])
    grad_y[1:-1, :] = np.abs(cur_i[2:, :] - cur_i[:-2, :])
    tex_img = grad_x + grad_y   # sum of abs central differences

    prev_i = prev.astype(np.int32)

    # ── sparse grid ─────────────────────────────────────────────────────────
    ys = list(range(y0, y1 + 1, GRID_STEP))
    xs = list(range(x0, x1 + 1, GRID_STEP))
    n_rows = len(ys)
    n_cols = len(xs)

    u_grid  = np.zeros((n_rows, n_cols), dtype=np.float32)
    v_grid  = np.zeros((n_rows, n_cols), dtype=np.float32)
    ok_grid = np.zeros((n_rows, n_cols), dtype=bool)

    flow_cnt = 0
    vx_sum   = 0.0
    vy_sum   = 0.0

    for ri, y in enumerate(ys):
        for ci, x in enumerate(xs):
            # ── texture gate ───────────────────────────────────────────────
            tex = int(
                tex_img[y - BLK_R: y + BLK_R + 1,
                        x - BLK_R: x + BLK_R + 1].sum()
            ) // (blk_sz * blk_sz)

            if tex < MIN_TEX:
                continue

            # ── block matching — vectorised over all search offsets ────────
            cur_blk = cur_i[y - BLK_R: y + BLK_R + 1,
                            x - BLK_R: x + BLK_R + 1]   # (blk_sz, blk_sz)

            # Sliding window over the prev search region.
            # Region covers every (dx,dy) offset in [-SRCH_R, +SRCH_R].
            ry = y - BLK_R - SRCH_R
            rx = x - BLK_R - SRCH_R
            region = prev_i[ry: ry + srch_n + blk_sz - 1,
                            rx: rx + srch_n + blk_sz - 1]   # (21, 21) for defaults

            if _HAVE_SWV:
                # shape (srch_n, srch_n, blk_sz, blk_sz)
                windows = _swv(region, (blk_sz, blk_sz))
            else:
                # manual as_strided fallback
                from numpy.lib.stride_tricks import as_strided
                s0, s1 = region.strides
                windows = as_strided(
                    region,
                    shape=(srch_n, srch_n, blk_sz, blk_sz),
                    strides=(s0, s1, s0, s1)
                )

            sad_map = np.abs(windows - cur_blk).sum(axis=(2, 3))  # (srch_n, srch_n)

            best_flat      = int(np.argmin(sad_map))
            best_di, best_dj = divmod(best_flat, srch_n)
            best_sad       = int(sad_map[best_di, best_dj])

            if best_sad > MAX_SAD:
                continue

            # flow: motion from prev→cur = -(offset that found the match)
            best_dx = best_dj - SRCH_R   # same as C++ bestDx
            best_dy = best_di - SRCH_R   # same as C++ bestDy
            u = float(-best_dx)          # u = -bestDx  (matches firmware)
            v = float(-best_dy)          # v = -bestDy

            ok_grid[ri, ci] = True
            u_grid[ri, ci]  = u
            v_grid[ri, ci]  = v
            vx_sum  += u
            vy_sum  += v
            flow_cnt += 1

    # ── divergence from finite-differences on grid ──────────────────────────
    div_sum   = 0.0;  div_cnt   = 0
    div_left  = 0.0;  div_right = 0.0
    left_cnt  = 0;    right_cnt = 0
    mid_col   = n_cols // 2

    for ri in range(n_rows):
        for ci in range(n_cols):
            if not ok_grid[ri, ci]:
                continue
            div_here = 0.0
            have     = False
            # du/dx  (right neighbour, same row)
            if ci + 1 < n_cols and ok_grid[ri, ci + 1]:
                div_here += float(u_grid[ri, ci + 1] - u_grid[ri, ci]) / GRID_STEP
                have = True
            # dv/dy  (row above — smaller y index)
            if ri > 0 and ok_grid[ri - 1, ci]:
                div_here += float(v_grid[ri, ci] - v_grid[ri - 1, ci]) / GRID_STEP
                have = True
            if have:
                div_sum += div_here
                div_cnt += 1
                if ci < mid_col:
                    div_left += div_here;  left_cnt  += 1
                else:
                    div_right += div_here; right_cnt += 1

    if div_cnt < MIN_DIV_CNT:
        return _empty_ofd()

    div_avg   = div_sum / div_cnt
    vx_mean   =  (vx_sum / flow_cnt) if flow_cnt > 0 else 0.0
    vy_mean   = -(vy_sum / flow_cnt) if flow_cnt > 0 else 0.0   # flip Y for roll=-180°
    tau       = (1.0 / div_avg) if div_avg > EPS_DIV else 1e6
    left_avg  = (div_left  / left_cnt)  if left_cnt  > 0 else 0.0
    right_avg = (div_right / right_cnt) if right_cnt > 0 else 0.0
    lr_balance = right_avg - left_avg

    return dict(divergence=div_avg, tau=tau, vx_mean=vx_mean, vy_mean=vy_mean,
                lr_balance=lr_balance, flow_cnt=flow_cnt, div_cnt=div_cnt, valid=1)


# ── incremental filter layer (mirrors ofd_task on Core 0) ─────────────────────
class OfdFilter:
    """Stateful per-frame OFD filter.  Mirrors the ofd_task() logic in firmware."""

    def __init__(self, params: dict):
        self.p = params
        self.ema_div          = 0.0
        self.ema_lr           = 0.0
        self._prev_ema_div    = 0.0
        self._initialized     = False
        self._prev_ts_ms      = None
        self._prev_tau_ms     = 0.0
        self._prev_turn_cmd   = 0.0

    def step(self, raw: dict, ts_ms: float, az: float) -> dict:
        """
        Process one frame.  raw = ofd_process_gray() result dict.
        Returns dict with filter output columns.
        """
        p = self.p
        az_quiet = int(abs(az - p["az_center"]) <= p["az_band"])
        _hold = dict(ema_div=self.ema_div, ema_lr=self.ema_lr,
                     tau_ms=self._prev_tau_ms, looming="0",
                     evasion_level="0", turn_cmd=self._prev_turn_cmd,
                     az_quiet=str(az_quiet))

        # Wing-sync gate
        if not az_quiet:
            _hold["az_quiet"] = "0"
            return _hold

        valid    = bool(int(raw.get("valid", 0)))
        div_cnt  = int(raw.get("div_cnt", 0))
        frame_ok = valid and div_cnt >= p["min_div_cnt"]

        if not frame_ok:
            return _hold

        # EMA update
        corrected = float(raw["divergence"]) - p["bias"]
        lr        = float(raw["lr_balance"])
        if not self._initialized:
            self.ema_div = corrected
            self.ema_lr  = lr
            self._initialized = True
        else:
            a = p["alpha"]
            self.ema_div = a * corrected        + (1.0 - a) * self.ema_div
            self.ema_lr  = a * lr               + (1.0 - a) * self.ema_lr

        # dt & tau
        if self._prev_ts_ms is not None:
            dt_ms = max(p["dt_min"], min(p["dt_max"], ts_ms - self._prev_ts_ms))
        else:
            dt_ms = p["dt_min"]
        self._prev_ts_ms = ts_ms

        dema = self.ema_div - self._prev_ema_div
        if abs(dema) < 1e-5:
            tau_ms = p["tau_max"]
        else:
            tau_s  = self.ema_div / (dema / (dt_ms / 1000.0))
            tau_ms = max(0.0, min(p["tau_max"], tau_s * 1000.0))   # neg → 0
        self._prev_ema_div = self.ema_div

        # Looming gate (div_cnt + valid also guard here)
        looming = (abs(self.ema_div) > p["div_thresh"]
                   and tau_ms < p["tau_brake"]
                   and frame_ok)

        evade = 0
        if looming:
            evade = 3 if tau_ms < p["tau_evade"] else 2

        turn_cmd = max(-1.0, min(1.0, -self.ema_lr * p["lr_gain"]))
        self._prev_tau_ms   = tau_ms
        self._prev_turn_cmd = turn_cmd

        return dict(ema_div=self.ema_div, ema_lr=self.ema_lr,
                    tau_ms=tau_ms, looming=str(int(looming)),
                    evasion_level=str(evade), turn_cmd=turn_cmd,
                    az_quiet="1")


# ── CSV loading ────────────────────────────────────────────────────────────────
def load_imu_csv(csv_path: str) -> tuple:
    """
    Load CSV as {frame_num: row_dict}.
    Returns (index_dict, has_filter_cols).
    """
    index      = {}
    has_filter = False
    try:
        with open(csv_path, newline="") as f:
            reader     = csv_mod.DictReader(f)
            fieldnames = reader.fieldnames or []
            has_filter = "ema_div" in fieldnames
            for row in reader:
                try:
                    index[int(row["frame"])] = row
                except (KeyError, ValueError):
                    pass
        schema = ("new (filter cols present)" if has_filter else "old (raw OFD only)")
        print(f"  CSV  : {len(index)} rows  |  schema: {schema}")
    except FileNotFoundError:
        print(f"  CSV  : not found ({os.path.basename(csv_path)})  — IMU fields will be 0")
    return index, has_filter


def _imu_defaults(az_center):
    return {"ax": "0.0", "ay": "0.0", "az": str(az_center),
            "gx": "0.0", "gy": "0.0", "gz": "0.0",
            "roll": "0.0", "pitch": "0.0", "yaw": "0.0"}


# ── panel row builder ──────────────────────────────────────────────────────────
def _build_row(frame_num: int, ts_ms: float,
               raw_ofd: dict, filt: dict, imu: dict) -> dict:
    """Merge OFD raw + filter + IMU into a panel-compatible row dict."""
    row = {
        "frame":         str(frame_num),
        "timestamp_ms":  str(ts_ms),
        "divergence":    str(raw_ofd.get("divergence",  0.0)),
        "tau":           str(raw_ofd.get("tau",         0.0)),
        "vx_mean":       str(raw_ofd.get("vx_mean",     0.0)),
        "vy_mean":       str(raw_ofd.get("vy_mean",     0.0)),
        "lr_balance":    str(raw_ofd.get("lr_balance",  0.0)),
        "flow_cnt":      str(raw_ofd.get("flow_cnt",    0)),
        "div_cnt":       str(raw_ofd.get("div_cnt",     0)),
        "valid":         str(raw_ofd.get("valid",       0)),
        # filter layer (stored as strings for consistency with panel reader)
        "ema_div":       str(filt.get("ema_div",       0.0)),
        "ema_lr":        str(filt.get("ema_lr",        0.0)),
        "tau_ms":        str(filt.get("tau_ms",        0.0)),
        "looming":       str(filt.get("looming",       "0")),
        "evasion_level": str(filt.get("evasion_level", "0")),
        "turn_cmd":      str(filt.get("turn_cmd",      0.0)),
        "az_quiet":      str(filt.get("az_quiet",      "0")),
    }
    # IMU
    for k in ("ax", "ay", "az", "gx", "gy", "gz", "roll", "pitch", "yaw"):
        row[k] = imu.get(k, "0.0")
    return row


# ── detection comparison ───────────────────────────────────────────────────────
def compare_detections(orig_index: dict, retuned_rows: list) -> dict:
    """Compare retuned filter output against original CSV filter cols."""
    total = len(retuned_rows)
    orig_loom = new_loom = gained = lost = changed_lvl = 0
    changed_frames = set()

    for row in retuned_rows:
        fn = int(row.get("frame", 0))
        orig = orig_index.get(fn, {})
        ol = int(orig.get("looming",        0))
        oe = int(orig.get("evasion_level",  0))
        rl = int(row.get("looming",         0))
        re = int(row.get("evasion_level",   0))

        orig_loom += ol
        new_loom  += rl

        if rl and not ol:
            gained += 1;      changed_frames.add(fn)
        elif ol and not rl:
            lost   += 1;      changed_frames.add(fn)
        elif ol and rl and oe != re:
            changed_lvl += 1; changed_frames.add(fn)

    return dict(total=total, orig_looming=orig_loom, new_looming=new_loom,
                gained=gained, lost=lost, changed_level=changed_lvl,
                changed_frames=changed_frames)


# ── output path ────────────────────────────────────────────────────────────────
def make_output_path(vid_path: str, params: dict, out_arg: str = None) -> str:
    if out_arg:
        return out_arg
    stem = os.path.splitext(vid_path)[0]
    ofd_w, ofd_h = params["ofd_res"]
    return (f"{stem}_retuned"
            f"_b{params['bias']}_a{params['alpha']}"
            f"_g{params['grid_step']}r{params['blk_r']}s{params['srch_r']}"
            f"_{ofd_w}x{ofd_h}.mp4")


# ── summary ────────────────────────────────────────────────────────────────────
def print_summary(params: dict, vid_path: str, out_path: str,
                  comparison: dict = None):
    ofd_w, ofd_h = params["ofd_res"]
    w = 60
    print(f"\n{'═'*w}")
    print(f"retune_ofd.py  —  {os.path.basename(vid_path)}")
    print(f"Output : {os.path.basename(out_path)}")
    print(f"{'─'*w}")
    print(f"OFD processing resolution  : {ofd_w}×{ofd_h}")
    print(f"Block-matching params:")
    print(f"  grid_step={params['grid_step']}  blk_r={params['blk_r']}  "
          f"srch_r={params['srch_r']}")
    print(f"  min_tex={params['min_tex']}  max_sad={params['max_sad']}  "
          f"eps_div={params['eps_div']:.0e}")
    print(f"Filter params:")
    print(f"  bias={params['bias']}  alpha={params['alpha']}  "
          f"div_thresh={params['div_thresh']:.3f}")
    print(f"  tau_evade={params['tau_evade']:.0f}ms  "
          f"tau_brake={params['tau_brake']:.0f}ms")
    print(f"  dt=[{params['dt_min']:.0f},{params['dt_max']:.0f}]ms  "
          f"az_center={params['az_center']:.3f}g  "
          f"az_band=\xb1{params['az_band']:.3f}g  "
          f"lr_gain={params['lr_gain']}")
    if comparison:
        print(f"{'─'*w}")
        print("Detection comparison vs original CSV filter columns:")
        print(f"  Original looming frames : {comparison['orig_looming']}")
        print(f"  Retuned  looming frames : {comparison['new_looming']}")
        print(f"  Gained detections       : {comparison['gained']}")
        print(f"  Lost detections         : {comparison['lost']}")
        print(f"  Changed evasion level   : {comparison['changed_level']}")
        print(f"  Changed frame count     : {len(comparison['changed_frames'])}")
    else:
        print(f"{'─'*w}")
        print("(no original filter cols in CSV — comparison unavailable)")
    print(f"{'═'*w}")


# ── main conversion ────────────────────────────────────────────────────────────
def convert_retuned(vid_path: str, csv_path: str, out_path: str,
                    params: dict) -> bool:
    print(f"\n{'─'*60}")
    print(f"Input  : {vid_path}")
    print(f"Output : {out_path}")

    ofd_w, ofd_h = params["ofd_res"]
    print(f"OFD res: {ofd_w}×{ofd_h}  |  grid_step={params['grid_step']}  "
          f"blk_r={params['blk_r']}  srch_r={params['srch_r']}")

    # ── load CSV (IMU reference) ───────────────────────────────────────────────
    imu_index, has_filter = load_imu_csv(csv_path)

    # ── read .vid header ───────────────────────────────────────────────────────
    with open(vid_path, "rb") as f:
        hdr_bytes = f.read(_cv.HEADER_SIZE)
    if len(hdr_bytes) < _cv.HEADER_SIZE:
        print("ERROR: file too small to contain a valid header.")
        return False

    frame_count, fps, width, height, frame_size = struct.unpack_from(
        _cv.HEADER_FMT, hdr_bytes)

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

    # ── set up video writer ────────────────────────────────────────────────────
    SEP_W = 2
    out_w = width + SEP_W + _cv.PANEL_W
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (out_w, height))
    if not writer.isOpened():
        print(f"ERROR: could not open output: {out_path}")
        return False

    print(f"Mode   : OFD retuned panel  ({out_w}×{height})")

    # ── per-frame processing ───────────────────────────────────────────────────
    ofd_filter   = OfdFilter(params)
    imu_defaults = _imu_defaults(params["az_center"])
    sep          = np.full((height, SEP_W, 3), (60, 60, 60), dtype=np.uint8)
    prev_gray    = None                     # OFD state: previous grayscale frame
    retuned_rows = []                       # for comparison against original CSV

    written = 0
    with open(vid_path, "rb") as f:
        f.seek(_cv.HEADER_SIZE)
        for i in range(frame_count):
            raw_bytes = f.read(frame_size)
            if len(raw_bytes) < frame_size:
                print(f"\n  WARNING: file truncated at frame {i + 1}")
                break

            frame_num = i + 1

            # ── decode frame ────────────────────────────────────────────────
            bgr = _cv.rgb565_to_bgr(raw_bytes, width, height)

            # ── grayscale at OFD resolution ──────────────────────────────────
            gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            if ofd_w != width or ofd_h != height:
                gray_ofd = cv2.resize(gray_full, (ofd_w, ofd_h),
                                      interpolation=cv2.INTER_AREA)
            else:
                gray_ofd = gray_full

            # ── block-matching OFD ───────────────────────────────────────────
            if prev_gray is None:
                raw_ofd = _empty_ofd()          # first frame: no prev available
            else:
                raw_ofd = ofd_process_gray(gray_ofd, prev_gray, params)
            prev_gray = gray_ofd.copy()

            # ── IMU + timestamp from CSV ─────────────────────────────────────
            csv_row = imu_index.get(frame_num, {})
            imu     = csv_row if csv_row else imu_defaults.copy()
            ts_ms   = float(csv_row.get("timestamp_ms",
                                        (frame_num - 1) * 1000.0 / fps))
            az      = float(imu.get("az", str(params["az_center"])))

            # ── filter layer ─────────────────────────────────────────────────
            filt = ofd_filter.step(raw_ofd, ts_ms, az)

            # ── build panel row ──────────────────────────────────────────────
            row = _build_row(frame_num, ts_ms, raw_ofd, filt, imu)
            retuned_rows.append(row)

            # ── render panel ─────────────────────────────────────────────────
            panel = _cv.build_ofd_panel(row, True, height)

            # override bottom hint
            cv2.rectangle(panel, (0, height - 14), (_cv.PANEL_W, height),
                          (30, 30, 30), -1)
            _cv._txt(panel, "retuned", 4, height - 4,
                     color=(80, 160, 255), scale=_cv.FS - 0.06)

            bgr_out = np.concatenate([bgr, sep, panel], axis=1)
            writer.write(bgr_out)
            written += 1

            if written % 30 == 0 or written == frame_count:
                pct = written / frame_count * 100
                bar = ("\u2588" * (written * 30 // frame_count)).ljust(30)
                print(f"  [{bar}] {written}/{frame_count} ({pct:.0f}%)",
                      end="\r")

    writer.release()
    print(f"\nDone   : {written} frames written \u2192 {out_path}")

    # ── post-process: comparison + orange stripe pass ─────────────────────────
    comparison = None
    changed_set = set()

    if has_filter and retuned_rows:
        comparison  = compare_detections(imu_index, retuned_rows)
        changed_set = comparison["changed_frames"]

    if changed_set:
        print(f"  Redrawing {len(changed_set)} changed frames with orange stripe…")
        _mark_changed_frames(out_path, retuned_rows, changed_set,
                             width, height, fps, params)

    print_summary(params, vid_path, out_path, comparison)
    return True


def _mark_changed_frames(mp4_path: str, rows: list, changed_set: set,
                         width: int, height: int, fps: int,
                         params: dict):
    """
    Re-open the MP4, add an orange left-stripe to changed frames, save.
    Uses a temp file to avoid in-place read/write conflicts.
    """
    import tempfile, shutil

    tmp = mp4_path + ".tmp.mp4"
    SEP_W  = 2
    out_w  = width + SEP_W + _cv.PANEL_W
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp, fourcc, float(fps), (out_w, height))
    if not writer.isOpened():
        print("  WARNING: could not reopen for stripe pass — skipping.")
        return

    cap = cv2.VideoCapture(mp4_path)
    row_index = {int(r["frame"]): r for r in rows}
    i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        i += 1
        if i in changed_set:
            frame[:, width + SEP_W: width + SEP_W + 3] = (0, 165, 255)  # orange stripe
        writer.write(frame)

    cap.release()
    writer.release()
    shutil.move(tmp, mp4_path)


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Re-run the full OFD pipeline from raw .VID frames with tunable "
            "block-matching and filter parameters, then export an annotated MP4."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("vid",
                    help=".VID file to process")
    ap.add_argument("--csv", default=None,
                    help="CSV sidecar for IMU data (auto-detected if omitted)")
    ap.add_argument("--out", "-o", default=None,
                    help="Output .mp4 path (auto-named if omitted)")

    # ── block-matching params ──────────────────────────────────────────────
    g1 = ap.add_argument_group(
        "Block-matching parameters (in-code defaults in CAPS at top of file)")
    g1.add_argument("--grid-step",  type=int,   default=OFD_GRID_STEP,
                    help=f"Grid spacing in OFD frame px (default: {OFD_GRID_STEP})")
    g1.add_argument("--blk-r",      type=int,   default=OFD_BLK_R,
                    help=f"Block radius → (2R+1)² patch (default: {OFD_BLK_R})")
    g1.add_argument("--srch-r",     type=int,   default=OFD_SRCH_R,
                    help=f"Search radius px in prev frame (default: {OFD_SRCH_R})")
    g1.add_argument("--min-tex",    type=int,   default=OFD_MIN_TEX,
                    help=f"Min texture score (default: {OFD_MIN_TEX})")
    g1.add_argument("--max-sad",    type=int,   default=OFD_MAX_SAD,
                    help=f"Max SAD to accept match (default: {OFD_MAX_SAD})")
    g1.add_argument("--eps-div",    type=float, default=OFD_EPS_DIV,
                    help=f"Epsilon for tau=1/div guard (default: {OFD_EPS_DIV:.0e})")
    g1.add_argument("--ofd-res",    default=OFD_RESOLUTION,
                    help=f"OFD processing resolution WxH (default: {OFD_RESOLUTION})")

    # ── filter params ──────────────────────────────────────────────────────
    g2 = ap.add_argument_group("Filter layer parameters")
    g2.add_argument("--bias",        type=float, default=OFD_DIV_BIAS,
                    help=f"Divergence DC bias (default: {OFD_DIV_BIAS})")
    g2.add_argument("--alpha",       type=float, default=OFD_EMA_ALPHA,
                    help=f"EMA smoothing factor 0–1 (default: {OFD_EMA_ALPHA})")
    g2.add_argument("--min-div-cnt", type=int,   default=OFD_MIN_DIV_CNT,
                    help=f"Min div_cnt for valid frame (default: {OFD_MIN_DIV_CNT})")
    g2.add_argument("--div-thresh",  type=float, default=OFD_DIV_THRESHOLD,
                    help=f"|ema_div| looming gate threshold (default: {OFD_DIV_THRESHOLD})")
    g2.add_argument("--tau-evade",   type=float, default=OFD_TAU_EVADE_MS,
                    help=f"EVADE threshold ms (default: {OFD_TAU_EVADE_MS})")
    g2.add_argument("--tau-brake",   type=float, default=OFD_TAU_BRAKE_MS,
                    help=f"BRAKE threshold ms (default: {OFD_TAU_BRAKE_MS})")
    g2.add_argument("--dt-min",      type=float, default=OFD_DT_MIN_MS,
                    help=f"Min dt clamp ms (default: {OFD_DT_MIN_MS})")
    g2.add_argument("--dt-max",      type=float, default=OFD_DT_MAX_MS,
                    help=f"Max dt clamp ms (default: {OFD_DT_MAX_MS})")
    g2.add_argument("--az-center",   type=float, default=OFD_AZ_QUIET_CENTER,
                    help=f"az quiet-zone center g (default: {OFD_AZ_QUIET_CENTER})")
    g2.add_argument("--az-band",     type=float, default=OFD_AZ_QUIET_BAND,
                    help=f"az quiet-zone ±band g (default: {OFD_AZ_QUIET_BAND})")
    g2.add_argument("--lr-gain",     type=float, default=OFD_LR_GAIN,
                    help=f"lr_balance→turn_cmd gain (default: {OFD_LR_GAIN})")

    args = ap.parse_args()

    if not os.path.isfile(args.vid):
        ap.error(f"VID file not found: {args.vid}")

    # parse resolution
    try:
        ofd_w, ofd_h = _parse_res(args.ofd_res)
    except argparse.ArgumentTypeError as e:
        ap.error(str(e))

    # auto-detect CSV
    csv_path = args.csv
    if csv_path is None:
        stem = os.path.splitext(args.vid)[0]
        csv_path = stem + ".CSV"
        if not os.path.exists(csv_path):
            csv_path = stem + ".csv"

    params = {
        # block-matching
        "grid_step":   args.grid_step,
        "blk_r":       args.blk_r,
        "srch_r":      args.srch_r,
        "min_tex":     args.min_tex,
        "max_sad":     args.max_sad,
        "eps_div":     args.eps_div,
        "ofd_res":     (ofd_w, ofd_h),
        # filter
        "bias":        args.bias,
        "alpha":       args.alpha,
        "min_div_cnt": args.min_div_cnt,
        "div_thresh":  args.div_thresh,
        "tau_evade":   args.tau_evade,
        "tau_brake":   args.tau_brake,
        "dt_min":      args.dt_min,
        "dt_max":      args.dt_max,
        "az_center":   args.az_center,
        "az_band":     args.az_band,
        "lr_gain":     args.lr_gain,
        "tau_max":     OFD_TAU_MAX,
    }

    out_path = make_output_path(args.vid, params, args.out)
    convert_retuned(args.vid, csv_path, out_path, params)


if __name__ == "__main__":
    main()
