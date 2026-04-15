#!/usr/bin/env python3
"""
viz_ofd.py  —  Optical Flow Vector Visualizer for OFD Parameter Tuning
========================================================================
Re-runs the block-matching OFD algorithm from recorded frames and draws
per-point flow vectors so you can visually tune every parameter.

Color key
---------
  Gray dot        — texture-rejected  (tex < MIN_TEX)
  Orange X        — SAD-rejected      (best SAD > MAX_SAD, but tex OK)
  Green→Yellow arrow — accepted flow  (green=low SAD, yellow=SAD near threshold)
  Red/blue tint   — divergence heatmap per cell (--show-div)
  Cyan circles    — search radius     (--show-search)
  White boxes     — block size        (--show-block)
  Cyan tint       — texture heatmap   (--show-tex)

Right panel
-----------
  Evasion badge   — NONE/ALERT/BRAKE/EVADE, color-coded
  Tau gauge       — log-scale bar, threshold markers
  ema_div bar     — centered ±0.15, threshold markers
  Turn arrow      — magnitude+direction of turn command
  IMU bars        — roll/pitch/yaw horizontal; ax/ay/az vertical
  AZ gate dot     — green=OPEN, red=SKIP

Keyboard (interactive mode)
---------------------------
  Space / →   advance one frame
  p           toggle auto-play
  s           save current frame as PNG
  r           reset OFD filter (clears EMA history)
  t           toggle right panel
  q / Esc     quit

  OFD Parameters window: drag trackbars to retune live.
  Filter resets automatically when any parameter changes.

Usage
-----
  python viz_ofd.py V0008.mp4
  python viz_ofd.py V0008.mp4 --min-tex 60 --max-sad 2000
  python viz_ofd.py V0008.mp4 --ofd-res 320x240 --scale 2
  python viz_ofd.py V0008.mp4 --show-div --show-search --save

Requirements
------------
  pip install opencv-python numpy
"""

import sys
import os
import argparse
import importlib.util
import pathlib
import math

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

# ── import utilities from convert_vid.py ─────────────────────────────────────
_HERE = pathlib.Path(__file__).parent
_cv_path = _HERE / "convert_vid.py"
if not _cv_path.exists():
    sys.exit(f"ERROR: convert_vid.py not found at {_cv_path}")
_spec = importlib.util.spec_from_file_location("convert_vid", _cv_path)
_cv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cv)

load_ofd      = _cv.load_ofd
_txt          = _cv._txt
FONT          = _cv.FONT

# ── layout constants ──────────────────────────────────────────────────────────
PANEL_W = 220          # baseline width of the right-side decision + IMU panel (px)
TB_WIN  = "OFD Parameters"   # trackbar window title

# ── default OFD / filter parameters ──────────────────────────────────────────
# Edit these directly for quick iteration without using CLI flags.
# All values mirror ofd_config.h defaults unless noted.
DEFAULTS = dict(
    # ── Block-matching ────────────────────────────────────────────────────────
    grid_step    = 4,      # px between flow-point grid nodes. ↓ = denser arrows, slower
    blk_r        = 4,       # block radius; block size = (2R+1)² = 9×9 px. ↑ = robust, less precise
    srch_r       = 6,       # max displacement search ±R px. ↑ = handles faster motion, O(R²) cost
    min_tex      = 25,      # min mean |∇| per block. ↑ = fewer but more reliable matches (gray dots)
    max_sad      = 4500,    # max SAD to accept a match. ↓ = stricter; rejected → orange X
    eps_div      = 1e-6,    # guard for tau = 1/div; prevents divide-by-zero

    # ── Filter layer ──────────────────────────────────────────────────────────
    div_bias     = -0.018,  # DC offset subtracted from raw divergence; calibrate from static hover
    ema_alpha    = 0.30,    # EMA weight on new sample. ↑ = faster response, more noise; range [0,1]
    min_div_cnt  = 10,      # min valid divergence terms per frame; frames below this → invalid
    div_threshold = 0.05,   # |ema_div| gate for looming detection (dual-gate outer condition)
    tau_evade    = 30.0,    # ms — EVADE triggered below this τ (most urgent, closest to impact)
    tau_brake    = 50.0,    # ms — BRAKE / dual-gate outer threshold; also looming condition
    dt_min       = 8.0,     # ms — clamp frame-interval floor; kills spurious short dt spikes
    dt_max       = 80.0,    # ms — clamp frame-interval ceiling; kills 1000ms+ dropout spikes

    # ── Wing-sync gate ────────────────────────────────────────────────────────
    az_center    = -0.986,  # g — gravity-only z-accel reference (quiet hover, no wing stroke)
    az_band      = 0.15,    # g — ±band; OFD skipped when az outside [center±band] (wing flap)

    # ── Steering ──────────────────────────────────────────────────────────────
    lr_gain      = 3.0,     # lr_balance → turn_cmd gain; tune so |turn_cmd| ≤ 1 in normal flight

    # ── Sentinel ──────────────────────────────────────────────────────────────
    tau_max      = 1000.0,  # ms — returned when no obstacle detected (τ bar full green)
)


class Mp4Source:
    """Random-access frame reader for MP4 input."""

    def __init__(self, path: str):
        self.path = path
        self.ext = pathlib.Path(path).suffix.lower()
        self.frame_count = 0
        self.fps = 10
        self.width = 0
        self.height = 0
        self._cap = None
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise ValueError("cannot open mp4")

        self.frame_count = max(0, int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        self.fps = int(round(self._cap.get(cv2.CAP_PROP_FPS) or 0)) or 10
        self.width = int(round(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0))
        self.height = int(round(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))

        if self.frame_count <= 0 or self.width <= 0 or self.height <= 0:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                raise ValueError("cannot read first frame")
            self.height, self.width = frame.shape[:2]
            if self.frame_count <= 0:
                self.frame_count = 1
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def read_frame(self, frame_num: int) -> np.ndarray | None:
        if frame_num < 1 or frame_num > self.frame_count:
            return None

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def resolve_csv_path(input_path: str) -> str | None:
    stem = pathlib.Path(input_path).with_suffix("")
    for suffix in (".CSV", ".csv"):
        candidate = stem.with_suffix(suffix)
        if candidate.exists():
            return str(candidate)
    return None


def compute_ui_scale(frame_w: int, frame_h: int) -> float:
    """Scale overlay text/panels with the displayed frame size."""
    scale = min(frame_w / 640.0, frame_h / 480.0)
    return max(1.0, min(2.2, scale))


def _csv_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _csv_int(row: dict, key: str, default: int = 0) -> int:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _csv_bool(row: dict, key: str, default: bool = False) -> bool:
    return bool(_csv_int(row, key, int(default)))


def build_csv_display_state(row: dict, raw_default: dict, filt_default: dict, params: dict) -> tuple[dict, dict]:
    """Use CSV telemetry for panel/log when available, otherwise fall back to computed values."""
    if not row:
        return raw_default, filt_default

    raw = dict(raw_default)
    filt = dict(filt_default)

    raw.update(
        divergence=_csv_float(row, "divergence", raw_default.get("divergence", 0.0)),
        tau=_csv_float(row, "tau", raw_default.get("tau", 0.0)),
        vx_mean=_csv_float(row, "vx_mean", raw_default.get("vx_mean", 0.0)),
        vy_mean=_csv_float(row, "vy_mean", raw_default.get("vy_mean", 0.0)),
        lr_balance=_csv_float(row, "lr_balance", raw_default.get("lr_balance", 0.0)),
        flow_cnt=_csv_int(row, "flow_cnt", raw_default.get("flow_cnt", 0)),
        div_cnt=_csv_int(row, "div_cnt", raw_default.get("div_cnt", 0)),
        valid=_csv_int(row, "valid", raw_default.get("valid", 0)),
        mean_flow_mag=_csv_float(row, "mean_flow_mag", raw_default.get("mean_flow_mag", 0.0)),
        mean_flow_mag_raw=_csv_float(row, "mean_flow_mag_raw", raw_default.get("mean_flow_mag_raw", 0.0)),
    )

    filt.update(
        ema_div=_csv_float(row, "ema_div", filt_default.get("ema_div", 0.0)),
        ema_lr=_csv_float(row, "ema_lr", filt_default.get("ema_lr", 0.0)),
        tau_ms=_csv_float(row, "tau_ms", filt_default.get("tau_ms", params["tau_max"])),
        looming=_csv_bool(row, "looming", filt_default.get("looming", False)),
        evasion_level=_csv_int(row, "evasion_level", filt_default.get("evasion_level", 0)),
        turn_cmd=_csv_float(row, "turn_cmd", filt_default.get("turn_cmd", 0.0)),
        az_quiet=_csv_bool(row, "az_quiet", filt_default.get("az_quiet", True)),
        ema_flow_mag=_csv_float(row, "ema_flow_mag", filt_default.get("ema_flow_mag", 0.0)),
    )

    return raw, filt


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK-MATCHING WITH PER-POINT DETAIL
# ══════════════════════════════════════════════════════════════════════════════

def ofd_process_gray_viz(cur: np.ndarray, prev: np.ndarray, params: dict):
    """
    Block-matching OFD — same algorithm as retune_ofd.py but also returns
    per-grid-point detail for visualization.

    Returns
    -------
    result : dict   — aggregate OFD output (same fields as firmware)
    points : list   — one dict per grid position:
        x, y        pixel coords in OFD frame
        tex         texture score
        tex_ok      bool — passed MIN_TEX
        dx, dy      best match offset (only if tex_ok)
        sad         best SAD value    (only if tex_ok, else -1)
        accepted    bool — passed MAX_SAD (only if tex_ok)
        div         local divergence contribution (float, nan if not computed)
        div_valid   bool — had at least one neighbor pair for divergence
    """
    gH, gW = cur.shape
    GRID_STEP   = params["grid_step"]
    BLK_R       = params["blk_r"]
    SRCH_R      = params["srch_r"]
    MIN_TEX     = params["min_tex"]
    MAX_SAD     = params["max_sad"]
    EPS_DIV     = params["eps_div"]
    MIN_DIV_CNT = params["min_div_cnt"]

    blk_sz = 2 * BLK_R + 1
    srch_n = 2 * SRCH_R + 1
    margin = BLK_R + SRCH_R + 2

    x0 = margin;  x1 = gW - margin - 1
    y0 = margin;  y1 = gH - margin - 1

    def _empty():
        return (dict(divergence=0.0, tau=0.0, vx_mean=0.0, vy_mean=0.0,
                     lr_balance=0.0, flow_cnt=0, div_cnt=0, valid=0), [])

    if x1 <= x0 or y1 <= y0:
        return _empty()

    # ── texture image ─────────────────────────────────────────────────────────
    cur_i  = cur.astype(np.int32)
    prev_i = prev.astype(np.int32)
    grad_x = np.zeros((gH, gW), np.int32)
    grad_y = np.zeros((gH, gW), np.int32)
    grad_x[:, 1:-1] = np.abs(cur_i[:, 2:] - cur_i[:, :-2])
    grad_y[1:-1, :] = np.abs(cur_i[2:, :] - cur_i[:-2, :])
    tex_img = (grad_x + grad_y)   # sum of abs central differences

    # ── sparse grid ───────────────────────────────────────────────────────────
    ys = list(range(y0, y1 + 1, GRID_STEP))
    xs = list(range(x0, x1 + 1, GRID_STEP))
    n_rows = len(ys)
    n_cols = len(xs)

    u_grid  = np.zeros((n_rows, n_cols), dtype=np.float32)
    v_grid  = np.zeros((n_rows, n_cols), dtype=np.float32)
    ok_grid = np.zeros((n_rows, n_cols), dtype=bool)

    # intermediate per-point storage (before divergence is known)
    pt_buf = []   # (ri, ci, x, y, tex, tex_ok, dx, dy, sad, accepted)

    flow_cnt = 0
    vx_sum   = 0.0
    vy_sum   = 0.0

    for ri, y in enumerate(ys):
        for ci, x in enumerate(xs):
            # ── texture gate ─────────────────────────────────────────────────
            tex_patch = tex_img[y - BLK_R: y + BLK_R + 1,
                                x - BLK_R: x + BLK_R + 1]
            tex = int(tex_patch.sum()) // (blk_sz * blk_sz)

            if tex < MIN_TEX:
                pt_buf.append((ri, ci, x, y, tex, False, 0, 0, -1, False))
                continue

            # ── block matching ───────────────────────────────────────────────
            cur_blk = cur_i[y - BLK_R: y + BLK_R + 1,
                            x - BLK_R: x + BLK_R + 1]
            ry = y - BLK_R - SRCH_R
            rx = x - BLK_R - SRCH_R
            region = prev_i[ry: ry + srch_n + blk_sz - 1,
                            rx: rx + srch_n + blk_sz - 1]

            if _HAVE_SWV:
                windows = _swv(region, (blk_sz, blk_sz))
            else:
                from numpy.lib.stride_tricks import as_strided
                s0, s1 = region.strides
                windows = as_strided(region,
                                     shape=(srch_n, srch_n, blk_sz, blk_sz),
                                     strides=(s0, s1, s0, s1))

            sad_map   = np.abs(windows - cur_blk).sum(axis=(2, 3))
            best_flat = int(np.argmin(sad_map))
            best_di, best_dj = divmod(best_flat, srch_n)
            best_sad  = int(sad_map[best_di, best_dj])

            if best_sad > MAX_SAD:
                pt_buf.append((ri, ci, x, y, tex, True,
                               best_dj - SRCH_R, best_di - SRCH_R, best_sad, False))
                continue

            best_dx = best_dj - SRCH_R
            best_dy = best_di - SRCH_R
            u = float(-best_dx)
            v = float(-best_dy)

            ok_grid[ri, ci] = True
            u_grid[ri, ci]  = u
            v_grid[ri, ci]  = v
            vx_sum  += u
            vy_sum  += v
            flow_cnt += 1

            pt_buf.append((ri, ci, x, y, tex, True, best_dx, best_dy, best_sad, True))

    # ── divergence finite-differences ────────────────────────────────────────
    div_sum   = 0.0;  div_cnt   = 0
    div_left  = 0.0;  div_right = 0.0
    left_cnt  = 0;    right_cnt = 0
    mid_col   = n_cols // 2

    # per-cell divergence array (indexed by [ri, ci])
    div_grid       = np.full((n_rows, n_cols), float("nan"))
    div_valid_grid = np.zeros((n_rows, n_cols), dtype=bool)

    for ri in range(n_rows):
        for ci in range(n_cols):
            if not ok_grid[ri, ci]:
                continue
            div_here = 0.0
            have     = False
            if ci + 1 < n_cols and ok_grid[ri, ci + 1]:
                div_here += float(u_grid[ri, ci + 1] - u_grid[ri, ci]) / GRID_STEP
                have = True
            if ri > 0 and ok_grid[ri - 1, ci]:
                div_here += float(v_grid[ri, ci] - v_grid[ri - 1, ci]) / GRID_STEP
                have = True
            if have:
                div_grid[ri, ci]       = div_here
                div_valid_grid[ri, ci] = True
                div_sum += div_here;  div_cnt += 1
                if ci < mid_col:
                    div_left  += div_here;  left_cnt  += 1
                else:
                    div_right += div_here;  right_cnt += 1

    # ── build points list ────────────────────────────────────────────────────
    points = []
    for (ri, ci, x, y, tex, tex_ok, dx, dy, sad, accepted) in pt_buf:
        d = float(div_grid[ri, ci]) if div_valid_grid[ri, ci] else float("nan")
        points.append(dict(
            x=x, y=y, ri=ri, ci=ci,
            tex=tex, tex_ok=tex_ok,
            dx=dx, dy=dy, sad=sad,
            accepted=accepted,
            div=d,
            div_valid=bool(div_valid_grid[ri, ci]),
        ))

    # ── aggregate result ─────────────────────────────────────────────────────
    if div_cnt < MIN_DIV_CNT:
        return (dict(divergence=0.0, tau=0.0, vx_mean=0.0, vy_mean=0.0,
                     lr_balance=0.0, flow_cnt=flow_cnt, div_cnt=div_cnt, valid=0),
                points)

    div_avg   = div_sum / div_cnt
    vx_mean   =  (vx_sum / flow_cnt) if flow_cnt > 0 else 0.0
    vy_mean   = -(vy_sum / flow_cnt) if flow_cnt > 0 else 0.0
    tau       = (1.0 / div_avg) if div_avg > EPS_DIV else 1e6
    left_avg  = (div_left  / left_cnt)  if left_cnt  > 0 else 0.0
    right_avg = (div_right / right_cnt) if right_cnt > 0 else 0.0
    lr_balance = right_avg - left_avg

    return (dict(divergence=div_avg, tau=tau, vx_mean=vx_mean, vy_mean=vy_mean,
                 lr_balance=lr_balance, flow_cnt=flow_cnt, div_cnt=div_cnt, valid=1),
            points)


# ══════════════════════════════════════════════════════════════════════════════
#  FILTER LAYER  (mirrors firmware ofd_task)
# ══════════════════════════════════════════════════════════════════════════════

class OfdFilter:
    def __init__(self, p):
        self.p            = p
        self.ema_div      = 0.0
        self.ema_lr       = 0.0
        self._prev_ts     = None
        self._prev_tau_ms = p["tau_max"]
        self._prev_turn   = 0.0

    def step(self, raw: dict, ts_ms: float, az: float) -> dict:
        p = self.p
        az_quiet = abs(az - p["az_center"]) <= p["az_band"]
        out = dict(ema_div=self.ema_div, ema_lr=self.ema_lr,
                   tau_ms=self._prev_tau_ms, looming=False,
                   evasion_level=0, turn_cmd=self._prev_turn,
                   az_quiet=az_quiet)

        if not raw["valid"] or not az_quiet:
            self._prev_ts = ts_ms
            return out

        dt = 0.0
        if self._prev_ts is not None:
            dt = max(p["dt_min"], min(p["dt_max"], ts_ms - self._prev_ts))
        self._prev_ts = ts_ms

        a = p["ema_alpha"]
        div_bc = raw["divergence"] - p["div_bias"]
        self.ema_div = a * div_bc + (1.0 - a) * self.ema_div
        self.ema_lr  = a * raw["lr_balance"] + (1.0 - a) * self.ema_lr

        tau_ms = p["tau_max"]
        if self.ema_div > p["eps_div"]:
            tau_ms = min(p["tau_max"], (1.0 / self.ema_div) * 1000.0)

        looming = (abs(self.ema_div) > p["div_threshold"] and
                   tau_ms < p["tau_brake"] and
                   raw["div_cnt"] >= p["min_div_cnt"])

        if tau_ms < p["tau_evade"]:
            evade = 3
        elif tau_ms < p["tau_brake"]:
            evade = 2
        elif tau_ms < 100.0:
            evade = 1
        else:
            evade = 0

        turn_cmd = max(-1.0, min(1.0, -self.ema_lr * p["lr_gain"]))

        self._prev_tau_ms = tau_ms
        self._prev_turn   = turn_cmd

        out.update(ema_div=self.ema_div, ema_lr=self.ema_lr, tau_ms=tau_ms,
                   looming=looming, evasion_level=evade, turn_cmd=turn_cmd,
                   az_quiet=az_quiet)
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# Colors (BGR)
COL_TEX_REJ  = (80,  80,  80)       # gray dot — texture rejected
COL_SAD_REJ  = (0,  120, 255)       # orange X — SAD rejected
COL_FLOW_LO  = (50, 220,  50)       # green  — low SAD (good match)
COL_FLOW_HI  = (50, 220, 220)       # yellow — SAD near threshold
COL_DIV_POS  = (0,   40, 200)       # red tint — positive divergence (looming)
COL_DIV_NEG  = (150,  40,  0)       # blue tint — negative divergence (receding)
COL_SEARCH   = (160, 160,  0)       # dim cyan — search radius
COL_BLOCK    = (220, 220, 220)      # white — block boundary
COL_LR_LINE  = (180, 180, 180)      # center L/R split line

EVADE_NAMES = {0: "NONE", 1: "ALERT", 2: "BRAKE", 3: "EVADE"}
EVADE_BADGE_COLS = {
    0: (30, 140,  30),   # green
    1: (0,  170, 170),   # cyan
    2: (0,  120, 230),   # orange
    3: (20,  20, 200),   # red
}
EVADE_TEXT_COLS = {
    0: (180, 255, 180),
    1: (200, 255, 255),
    2: (180, 210, 255),
    3: (200, 180, 255),
}


def _lerp_color(c0, c1, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c0[i] + t * (c1[i] - c0[i])) for i in range(3))


def _draw_arrow(img, x0, y0, dx, dy, color, scale=1):
    x1 = int(x0 + dx * scale)
    y1 = int(y0 + dy * scale)
    cv2.arrowedLine(img, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA, tipLength=0.35)


def _draw_x(img, cx, cy, half, color, thickness=1):
    cv2.line(img, (cx - half, cy - half), (cx + half, cy + half),
             color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx + half, cy - half), (cx - half, cy + half),
             color, thickness, cv2.LINE_AA)


def _draw_hbar(img, x, y, w, h, value, vmin, vmax,
               col_fill=(0, 200, 0), col_bg=(40, 40, 40), col_border=(80, 80, 80),
               center=False):
    """Horizontal bar gauge. If center=True, bar grows left/right from center."""
    cv2.rectangle(img, (x, y), (x + w, y + h), col_bg, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), col_border, 1)
    span = max(vmax - vmin, 1e-9)
    t = (value - vmin) / span
    t = max(0.0, min(1.0, t))
    if center:
        mid = x + w // 2
        cx2 = x + int(t * w)
        x1, x2 = (cx2, mid) if cx2 < mid else (mid, cx2)
        if x2 > x1:
            cv2.rectangle(img, (x1 + 1, y + 1), (x2 - 1, y + h - 1), col_fill, -1)
    else:
        fw = int(t * w)
        if fw > 1:
            cv2.rectangle(img, (x + 1, y + 1), (x + fw, y + h - 1), col_fill, -1)


def _draw_vbar(img, x, y, w, h, value, vmin, vmax,
               col_fill=(0, 200, 0), col_bg=(40, 40, 40), col_border=(80, 80, 80),
               center=False):
    """Vertical bar gauge. If center=True, bar grows up/down from mid."""
    cv2.rectangle(img, (x, y), (x + w, y + h), col_bg, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), col_border, 1)
    span = max(vmax - vmin, 1e-9)
    t = (value - vmin) / span
    t = max(0.0, min(1.0, t))
    if center:
        mid = y + h // 2
        cy2 = y + h - int(t * h)
        y1, y2 = (cy2, mid) if cy2 < mid else (mid, cy2)
        if y2 > y1:
            cv2.rectangle(img, (x + 1, y1 + 1), (x + w - 1, y2 - 1), col_fill, -1)
    else:
        fh = int(t * h)
        if fh > 1:
            cv2.rectangle(img, (x + 1, y + h - fh), (x + w - 1, y + h - 1), col_fill, -1)


# ══════════════════════════════════════════════════════════════════════════════
#  RIGHT-SIDE DECISION + IMU PANEL
# ══════════════════════════════════════════════════════════════════════════════

def draw_decision_panel(panel: np.ndarray, filt: dict, imu_row: dict, params: dict):
    """
    Draw decision + IMU information onto a pre-allocated panel slice.
    panel : ndarray (h, PANEL_W, 3) — will be drawn in-place
    """
    ph, pw = panel.shape[:2]
    panel[:] = (22, 22, 22)

    lvl      = int(filt.get("evasion_level", 0))
    ema_div  = filt.get("ema_div",  0.0)
    ema_lr   = filt.get("ema_lr",   0.0)
    tau_ms   = filt.get("tau_ms",   params["tau_max"])
    turn_cmd = filt.get("turn_cmd", 0.0)
    az_quiet = filt.get("az_quiet", True)
    looming  = filt.get("looming",  False)

    M = 4   # margin

    # ── evasion badge ─────────────────────────────────────────────────────────
    badge_col = EVADE_BADGE_COLS[lvl]
    badge_txt_col = EVADE_TEXT_COLS[lvl]
    by = M
    bh = 36
    cv2.rectangle(panel, (M, by), (pw - M, by + bh), badge_col, -1)
    if looming:
        cv2.rectangle(panel, (M, by), (pw - M, by + bh), (0, 0, 255), 2)
    else:
        cv2.rectangle(panel, (M, by), (pw - M, by + bh), (100, 100, 100), 1)
    name = EVADE_NAMES[lvl]
    (tw, th), _ = cv2.getTextSize(name, FONT, 0.65, 2)
    cv2.putText(panel, name,
                ((pw - tw) // 2, by + bh // 2 + th // 2),
                FONT, 0.65, badge_txt_col, 2, cv2.LINE_AA)

    cy = by + bh + 6

    # ── tau gauge (log scale 10..1000 ms) ────────────────────────────────────
    _txt(panel, f"tau {tau_ms:.0f} ms", M, cy + 9, scale=0.33, color=(200, 200, 200))
    cy += 12
    bw = pw - 2 * M
    cv2.rectangle(panel, (M, cy), (M + bw, cy + 11), (40, 40, 40), -1)

    # log-scale fill: high tau = full bar = far from obstacle → green
    t_log = (math.log10(max(10.0, min(1000.0, tau_ms))) - 1.0) / 2.0
    fw = int(t_log * bw)
    bar_col = EVADE_BADGE_COLS[lvl]
    if fw > 1:
        cv2.rectangle(panel, (M + 1, cy + 1), (M + fw, cy + 10), bar_col, -1)

    # threshold markers (log-scaled)
    def _log_x(ms):
        t = (math.log10(max(10.0, min(1000.0, ms))) - 1.0) / 2.0
        return M + int(t * bw)

    mx_brake = _log_x(params["tau_brake"])
    mx_evade = _log_x(params["tau_evade"])
    cv2.line(panel, (mx_brake, cy), (mx_brake, cy + 11), (0, 140, 255), 1)
    cv2.line(panel, (mx_evade, cy), (mx_evade, cy + 11), (0,  40, 220), 1)
    cv2.rectangle(panel, (M, cy), (M + bw, cy + 11), (90, 90, 90), 1)
    cy += 15

    # ── ema_div bar (centered, ±0.15) ────────────────────────────────────────
    _txt(panel, f"ema_div {ema_div:+.4f}", M, cy + 9, scale=0.33, color=(200, 200, 200))
    cy += 12
    div_col = (40, 80, 210) if ema_div >= 0 else (160, 80, 30)
    _draw_hbar(panel, M, cy, bw, 11, ema_div, -0.15, 0.15,
               col_fill=div_col, center=True)
    # threshold markers ±div_threshold
    half_bw = bw // 2
    dt_px = int(params["div_threshold"] / 0.15 * half_bw)
    mid_x = M + half_bw
    cv2.line(panel, (mid_x + dt_px, cy), (mid_x + dt_px, cy + 11), (0, 220, 220), 1)
    cv2.line(panel, (mid_x - dt_px, cy), (mid_x - dt_px, cy + 11), (0, 220, 220), 1)
    cv2.line(panel, (mid_x, cy), (mid_x, cy + 11), (60, 60, 60), 1)
    cy += 15

    # ── lr_balance bar (centered, ±0.10) ─────────────────────────────────────
    _txt(panel, f"lr_bal {ema_lr:+.4f}", M, cy + 9, scale=0.33, color=(180, 180, 180))
    cy += 12
    _draw_hbar(panel, M, cy, bw, 9, ema_lr, -0.10, 0.10,
               col_fill=(180, 140, 30), center=True)
    cv2.line(panel, (mid_x, cy), (mid_x, cy + 9), (60, 60, 60), 1)
    cy += 14

    # ── turn command arrow ────────────────────────────────────────────────────
    _txt(panel, f"turn  {turn_cmd:+.3f}", M, cy + 9, scale=0.33, color=(200, 200, 200))
    cy += 12
    arrow_y = cy + 7
    cv2.line(panel, (M, arrow_y), (M + bw, arrow_y), (55, 55, 55), 1)
    arrow_len = int(abs(turn_cmd) * (bw // 2 - 2))
    center_x  = M + bw // 2
    if arrow_len > 2:
        arrow_col = (0, 220, 220)
        if turn_cmd < 0:  # turn left
            cv2.arrowedLine(panel, (center_x, arrow_y),
                            (center_x - arrow_len, arrow_y),
                            arrow_col, 2, tipLength=0.35)
        else:             # turn right
            cv2.arrowedLine(panel, (center_x, arrow_y),
                            (center_x + arrow_len, arrow_y),
                            arrow_col, 2, tipLength=0.35)
    cy += 18

    # ── az_quiet gate indicator ───────────────────────────────────────────────
    gate_col = (40, 200, 40) if az_quiet else (30, 30, 200)
    gate_txt = "GATE:OPEN" if az_quiet else "GATE:SKIP"
    cv2.circle(panel, (M + 5, cy + 5), 5, gate_col, -1)
    _txt(panel, gate_txt, M + 14, cy + 9, color=gate_col, scale=0.33)
    cy += 16

    # ── separator ─────────────────────────────────────────────────────────────
    cv2.line(panel, (0, cy), (pw, cy), (70, 70, 70), 1)
    _txt(panel, "IMU", pw // 2 - 10, cy + 10, scale=0.33, color=(130, 130, 130))
    cy += 14

    # ── IMU data ──────────────────────────────────────────────────────────────
    ax    = imu_row.get("ax",    0.0)
    ay    = imu_row.get("ay",    0.0)
    az    = imu_row.get("az",    0.0)
    gx    = imu_row.get("gx",   0.0)
    gy    = imu_row.get("gy",   0.0)
    gz    = imu_row.get("gz",   0.0)
    roll  = imu_row.get("roll",  0.0)
    pitch = imu_row.get("pitch", 0.0)
    yaw   = imu_row.get("yaw",   0.0)

    # roll / pitch / yaw horizontal bars (centered, ±180°)
    angle_items = [
        ("roll",  roll,  -180.0, 180.0, (100, 200, 255)),
        ("pitch", pitch,  -90.0,  90.0, (100, 255, 180)),
        ("yaw",   yaw,   -180.0, 180.0, (255, 200, 100)),
    ]
    for label, val, lo, hi, col in angle_items:
        _txt(panel, f"{label} {val:+.0f}°", M, cy + 9, scale=0.32, color=(180, 180, 180))
        cy += 12
        _draw_hbar(panel, M, cy, bw, 9, val, lo, hi,
                   col_fill=col, center=True)
        cv2.line(panel, (M + bw // 2, cy), (M + bw // 2, cy + 9), (55, 55, 55), 1)
        cy += 12

    # accel ax / ay / az — vertical bars (centered at 0, ±2g)
    cy += 2
    _txt(panel, f"ax{ax:+.2f} ay{ay:+.2f} az{az:+.2f}", M, cy + 8,
         scale=0.28, color=(160, 160, 160))
    cy += 10
    remaining = ph - cy - M
    bar_h_v = max(10, min(50, remaining - 12))
    bar_w_v = (bw - 2 * 2) // 3   # 3 columns with 2px gap
    accel_items = [("ax", ax, (100, 180, 255)),
                   ("ay", ay, (255, 180, 100)),
                   ("az", az, (100, 255, 180))]
    for i, (label, val, col) in enumerate(accel_items):
        bx = M + i * (bar_w_v + 2)
        _draw_vbar(panel, bx, cy, bar_w_v, bar_h_v, val, -2.0, 2.0,
                   col_fill=col, center=True)
        cv2.line(panel, (bx, cy + bar_h_v // 2),
                 (bx + bar_w_v, cy + bar_h_v // 2), (55, 55, 55), 1)
        _txt(panel, label, bx, cy + bar_h_v + 9, scale=0.27, color=(140, 140, 140))

    # gyro text (compact, bottom of panel)
    gy_y = cy + bar_h_v + 14
    if gy_y + 9 < ph:
        _txt(panel, f"gx{gx:+.0f} gy{gy:+.0f} gz{gz:+.0f}",
             M, gy_y + 8, scale=0.27, color=(120, 120, 120))


# ══════════════════════════════════════════════════════════════════════════════
#  FLOW OVERLAY + STATS
# ══════════════════════════════════════════════════════════════════════════════

def draw_flow_overlay(vis: np.ndarray, points: list, params: dict,
                      scale: int, ofd_w: int, ofd_h: int,
                      frame_w: int, frame_h: int,
                      opts: dict):
    """
    Draw per-point flow vectors onto vis (which is the upscaled BGR frame).
    coords in 'points' are in OFD space; we scale up to vis space.
    """
    sx = frame_w / ofd_w   # OFD-pixel → display-pixel scale X
    sy = frame_h / ofd_h   # OFD-pixel → display-pixel scale Y
    MAX_SAD   = params["max_sad"]
    SRCH_R    = params["srch_r"]
    BLK_R     = params["blk_r"]
    GRID_STEP = params["grid_step"]

    for pt in points:
        px = int(pt["x"] * sx)
        py = int(pt["y"] * sy)

        # ── optional: texture heatmap tint ───────────────────────────────────
        if opts.get("show_tex") and pt["tex_ok"]:
            norm = min(1.0, pt["tex"] / 200.0)
            half = max(1, int(GRID_STEP * sx / 2))
            overlay_col = (int(50 * norm), int(50 * norm), 0)
            cv2.rectangle(vis,
                          (px - half, py - half),
                          (px + half, py + half),
                          overlay_col, -1)

        # ── optional: divergence heatmap ─────────────────────────────────────
        if opts.get("show_div") and pt["div_valid"] and not math.isnan(pt["div"]):
            half = max(1, int(GRID_STEP * sx / 2))
            alpha = min(1.0, abs(pt["div"]) / 0.1)
            base = COL_DIV_POS if pt["div"] > 0 else COL_DIV_NEG
            tint = tuple(int(b * alpha) for b in base)
            sub = vis[py - half: py + half, px - half: px + half]
            if sub.size:
                blended = np.clip(sub.astype(np.int32) + np.array(tint, np.int32),
                                  0, 255).astype(np.uint8)
                vis[py - half: py + half, px - half: px + half] = blended

        # ── optional: search radius circle ───────────────────────────────────
        if opts.get("show_search"):
            r = int(SRCH_R * min(sx, sy))
            cv2.circle(vis, (px, py), r, COL_SEARCH, 1, cv2.LINE_AA)

        # ── optional: block-size box ──────────────────────────────────────────
        if opts.get("show_block"):
            half = max(1, int(BLK_R * min(sx, sy)))
            cv2.rectangle(vis, (px - half, py - half), (px + half, py + half),
                          COL_BLOCK, 1)

        # ── per-point marker ─────────────────────────────────────────────────
        if not pt["tex_ok"]:
            cv2.circle(vis, (px, py), 2, COL_TEX_REJ, -1)

        elif not pt["accepted"]:
            half = max(3, int(3 * min(sx, sy)))
            _draw_x(vis, px, py, half, COL_SAD_REJ, 1)

        else:
            t = pt["sad"] / MAX_SAD
            col = _lerp_color(COL_FLOW_LO, COL_FLOW_HI, t)
            u = -pt["dx"]
            v = -pt["dy"]
            arrow_scale = min(sx, sy) * 2.5
            _draw_arrow(vis, px, py, u, v, col, scale=arrow_scale)
            cv2.circle(vis, (px, py), 2, col, -1)

    # ── L/R split line ────────────────────────────────────────────────────────
    mid_x = frame_w // 2
    cv2.line(vis, (mid_x, 0), (mid_x, frame_h), COL_LR_LINE, 1)


def draw_stats_overlay(vis: np.ndarray, raw: dict, filt: dict,
                       frame_num: int, params: dict, frame_h: int):
    """Draw stats text at the bottom of the frame."""
    div     = raw.get("divergence", 0.0)
    tau_raw = raw.get("tau", 0.0)
    fcnt    = raw.get("flow_cnt", 0)
    dcnt    = raw.get("div_cnt", 0)
    valid   = raw.get("valid", 0)

    ema     = filt.get("ema_div", 0.0)
    tau_ms  = filt.get("tau_ms", params["tau_max"])
    lvl     = int(filt.get("evasion_level", 0))
    az_q    = filt.get("az_quiet", True)
    turn    = filt.get("turn_cmd", 0.0)

    col_lvl  = EVADE_TEXT_COLS[lvl]
    col_val  = (220, 220, 220) if valid else (80, 80, 200)
    col_gate = (50, 220, 50) if az_q else (80, 80, 180)

    lines = [
        (f"F:{frame_num:04d}  div:{div:+.4f}  tau:{tau_raw*1000:.0f}ms"
         f"  flow:{fcnt}  div_cnt:{dcnt}",       col_val),
        (f"ema:{ema:+.4f}  tau_ms:{tau_ms:.0f}"
         f"  lvl:{EVADE_NAMES[lvl]}  turn:{turn:+.3f}", col_lvl),
        (f"gate:{'OPEN' if az_q else 'SKIP'}"
         f"  GRID:{params['grid_step']}"
         f"  BLK_R:{params['blk_r']}"
         f"  SRCH_R:{params['srch_r']}"
         f"  TEX:{params['min_tex']}"
         f"  SAD:{params['max_sad']}",            col_gate),
    ]

    LH = 14
    y0 = frame_h - LH * len(lines) - 4
    for i, (txt, col) in enumerate(lines):
        _txt(vis, txt, 4, y0 + i * LH + 10, color=col, scale=0.36, thickness=1)


def draw_decision_panel(panel: np.ndarray, filt: dict, imu_row: dict, params: dict, ui_scale: float = 1.0):
    """Scaled right-side decision + IMU panel."""
    ph, pw = panel.shape[:2]
    panel[:] = (22, 22, 22)

    lvl      = int(filt.get("evasion_level", 0))
    ema_div  = filt.get("ema_div",  0.0)
    ema_lr   = filt.get("ema_lr",   0.0)
    tau_ms   = filt.get("tau_ms",   params["tau_max"])
    turn_cmd = filt.get("turn_cmd", 0.0)
    az_quiet = filt.get("az_quiet", True)
    looming  = filt.get("looming",  False)

    M = max(4, int(round(4 * ui_scale)))
    txt_scale = 0.33 * ui_scale
    txt_thick = max(1, int(round(ui_scale)))
    badge_scale = 0.65 * ui_scale
    badge_thick = max(2, int(round(2 * ui_scale)))
    bar_h = max(11, int(round(11 * ui_scale)))
    small_bar_h = max(9, int(round(9 * ui_scale)))

    badge_col = EVADE_BADGE_COLS[lvl]
    by = M
    bh = max(36, int(round(36 * ui_scale)))
    cv2.rectangle(panel, (M, by), (pw - M, by + bh), badge_col, -1)
    cv2.rectangle(panel, (M, by), (pw - M, by + bh), (0, 0, 255) if looming else (100, 100, 100),
                  badge_thick if looming else 1)
    name = EVADE_NAMES[lvl]
    (tw, th), _ = cv2.getTextSize(name, FONT, badge_scale, badge_thick)
    cv2.putText(panel, name, ((pw - tw) // 2, by + bh // 2 + th // 2),
                FONT, badge_scale, EVADE_TEXT_COLS[lvl], badge_thick, cv2.LINE_AA)

    cy = by + bh + max(6, int(round(6 * ui_scale)))
    bw = pw - 2 * M
    _txt(panel, f"tau {tau_ms:.0f} ms", M, cy + int(round(9 * ui_scale)),
         scale=txt_scale, thickness=txt_thick, color=(200, 200, 200))
    cy += max(12, int(round(12 * ui_scale)))
    cv2.rectangle(panel, (M, cy), (M + bw, cy + bar_h), (40, 40, 40), -1)
    t_log = (math.log10(max(10.0, min(1000.0, tau_ms))) - 1.0) / 2.0
    fw = int(t_log * bw)
    if fw > 1:
        cv2.rectangle(panel, (M + 1, cy + 1), (M + fw, cy + bar_h - 1), badge_col, -1)

    def _log_x(ms):
        t = (math.log10(max(10.0, min(1000.0, ms))) - 1.0) / 2.0
        return M + int(t * bw)

    cv2.line(panel, (_log_x(params["tau_brake"]), cy), (_log_x(params["tau_brake"]), cy + bar_h), (0, 140, 255), txt_thick)
    cv2.line(panel, (_log_x(params["tau_evade"]), cy), (_log_x(params["tau_evade"]), cy + bar_h), (0, 40, 220), txt_thick)
    cv2.rectangle(panel, (M, cy), (M + bw, cy + bar_h), (90, 90, 90), 1)
    cy += max(15, int(round(15 * ui_scale)))

    _txt(panel, f"ema_div {ema_div:+.4f}", M, cy + int(round(9 * ui_scale)),
         scale=txt_scale, thickness=txt_thick, color=(200, 200, 200))
    cy += max(12, int(round(12 * ui_scale)))
    div_col = (40, 80, 210) if ema_div >= 0 else (160, 80, 30)
    _draw_hbar(panel, M, cy, bw, bar_h, ema_div, -0.15, 0.15, col_fill=div_col, center=True)
    half_bw = bw // 2
    dt_px = int(params["div_threshold"] / 0.15 * half_bw)
    mid_x = M + half_bw
    cv2.line(panel, (mid_x + dt_px, cy), (mid_x + dt_px, cy + bar_h), (0, 220, 220), txt_thick)
    cv2.line(panel, (mid_x - dt_px, cy), (mid_x - dt_px, cy + bar_h), (0, 220, 220), txt_thick)
    cv2.line(panel, (mid_x, cy), (mid_x, cy + bar_h), (60, 60, 60), 1)
    cy += max(15, int(round(15 * ui_scale)))

    _txt(panel, f"lr_bal {ema_lr:+.4f}", M, cy + int(round(9 * ui_scale)),
         scale=txt_scale, thickness=txt_thick, color=(180, 180, 180))
    cy += max(12, int(round(12 * ui_scale)))
    _draw_hbar(panel, M, cy, bw, small_bar_h, ema_lr, -0.10, 0.10, col_fill=(180, 140, 30), center=True)
    cv2.line(panel, (mid_x, cy), (mid_x, cy + small_bar_h), (60, 60, 60), 1)
    cy += max(14, int(round(14 * ui_scale)))

    _txt(panel, f"turn  {turn_cmd:+.3f}", M, cy + int(round(9 * ui_scale)),
         scale=txt_scale, thickness=txt_thick, color=(200, 200, 200))
    cy += max(12, int(round(12 * ui_scale)))
    arrow_y = cy + max(7, int(round(7 * ui_scale)))
    cv2.line(panel, (M, arrow_y), (M + bw, arrow_y), (55, 55, 55), 1)
    arrow_len = int(abs(turn_cmd) * (bw // 2 - 2))
    center_x = M + bw // 2
    if arrow_len > 2:
        end_x = center_x - arrow_len if turn_cmd < 0 else center_x + arrow_len
        cv2.arrowedLine(panel, (center_x, arrow_y), (end_x, arrow_y), (0, 220, 220), max(2, txt_thick), tipLength=0.35)
    cy += max(18, int(round(18 * ui_scale)))

    gate_col = (40, 200, 40) if az_quiet else (30, 30, 200)
    gate_txt = "GATE:OPEN" if az_quiet else "GATE:SKIP"
    dot_r = max(5, int(round(5 * ui_scale)))
    cv2.circle(panel, (M + dot_r, cy + dot_r), dot_r, gate_col, -1)
    _txt(panel, gate_txt, M + max(14, int(round(14 * ui_scale))), cy + int(round(9 * ui_scale)),
         color=gate_col, scale=txt_scale, thickness=txt_thick)
    cy += max(16, int(round(16 * ui_scale)))

    cv2.line(panel, (0, cy), (pw, cy), (70, 70, 70), 1)
    _txt(panel, "IMU", pw // 2 - int(round(10 * ui_scale)), cy + int(round(10 * ui_scale)),
         scale=txt_scale, thickness=txt_thick, color=(130, 130, 130))
    cy += max(14, int(round(14 * ui_scale)))

    ax = imu_row.get("ax", 0.0); ay = imu_row.get("ay", 0.0); az = imu_row.get("az", 0.0)
    gx = imu_row.get("gx", 0.0); gy = imu_row.get("gy", 0.0); gz = imu_row.get("gz", 0.0)
    roll = imu_row.get("roll", 0.0); pitch = imu_row.get("pitch", 0.0); yaw = imu_row.get("yaw", 0.0)

    for label, val, lo, hi, col in [
        ("roll", roll, -180.0, 180.0, (100, 200, 255)),
        ("pitch", pitch, -90.0, 90.0, (100, 255, 180)),
        ("yaw", yaw, -180.0, 180.0, (255, 200, 100)),
    ]:
        _txt(panel, f"{label} {val:+.0f}°", M, cy + int(round(9 * ui_scale)),
             scale=0.32 * ui_scale, thickness=txt_thick, color=(180, 180, 180))
        cy += max(12, int(round(12 * ui_scale)))
        _draw_hbar(panel, M, cy, bw, small_bar_h, val, lo, hi, col_fill=col, center=True)
        cv2.line(panel, (M + bw // 2, cy), (M + bw // 2, cy + small_bar_h), (55, 55, 55), 1)
        cy += max(12, int(round(12 * ui_scale)))

    cy += max(2, int(round(2 * ui_scale)))
    _txt(panel, f"ax{ax:+.2f} ay{ay:+.2f} az{az:+.2f}", M, cy + int(round(8 * ui_scale)),
         scale=0.28 * ui_scale, thickness=txt_thick, color=(160, 160, 160))
    cy += max(10, int(round(10 * ui_scale)))
    remaining = ph - cy - M
    bar_h_v = max(int(round(16 * ui_scale)), min(int(round(80 * ui_scale)), remaining - max(12, int(round(12 * ui_scale)))))
    gap_v = max(2, int(round(2 * ui_scale)))
    bar_w_v = max(8, (bw - 2 * gap_v) // 3)
    for i, (label, val, col) in enumerate([("ax", ax, (100, 180, 255)), ("ay", ay, (255, 180, 100)), ("az", az, (100, 255, 180))]):
        bx = M + i * (bar_w_v + gap_v)
        _draw_vbar(panel, bx, cy, bar_w_v, bar_h_v, val, -2.0, 2.0, col_fill=col, center=True)
        cv2.line(panel, (bx, cy + bar_h_v // 2), (bx + bar_w_v, cy + bar_h_v // 2), (55, 55, 55), 1)
        _txt(panel, label, bx, cy + bar_h_v + int(round(9 * ui_scale)),
             scale=0.27 * ui_scale, thickness=txt_thick, color=(140, 140, 140))

    gy_y = cy + bar_h_v + max(14, int(round(14 * ui_scale)))
    if gy_y + int(round(9 * ui_scale)) < ph:
        _txt(panel, f"gx{gx:+.0f} gy{gy:+.0f} gz{gz:+.0f}",
             M, gy_y + int(round(8 * ui_scale)),
             scale=0.27 * ui_scale, thickness=txt_thick, color=(120, 120, 120))


def draw_stats_overlay(vis: np.ndarray, raw: dict, filt: dict,
                       frame_num: int, params: dict, frame_h: int, ui_scale: float = 1.0):
    """Scaled stats text at the bottom-left of the frame."""
    div     = raw.get("divergence", 0.0)
    tau_raw = raw.get("tau", 0.0)
    fcnt    = raw.get("flow_cnt", 0)
    dcnt    = raw.get("div_cnt", 0)
    valid   = raw.get("valid", 0)
    mag     = raw.get("mean_flow_mag", 0.0)

    ema     = filt.get("ema_div", 0.0)
    ema_mag = filt.get("ema_flow_mag", 0.0)
    tau_ms  = filt.get("tau_ms", params["tau_max"])
    lvl     = int(filt.get("evasion_level", 0))
    az_q    = filt.get("az_quiet", True)
    turn    = filt.get("turn_cmd", 0.0)
    looming = filt.get("looming", False)

    col_lvl  = EVADE_TEXT_COLS[lvl]
    col_val  = (220, 220, 220) if valid else (80, 80, 200)
    col_gate = (50, 220, 50) if az_q else (80, 80, 180)

    lines = [
        (f"F:{frame_num:04d}  div:{div:+.4f}  tau:{tau_raw*1000:.0f}ms  flow:{fcnt}  div_cnt:{dcnt}  mag:{mag:.3f}", col_val),
        (f"ema:{ema:+.4f}  tau_ms:{tau_ms:.0f}  lvl:{EVADE_NAMES[lvl]}  looming:{int(bool(looming))}  turn:{turn:+.3f}  ema_mag:{ema_mag:.3f}", col_lvl),
        (f"gate:{'OPEN' if az_q else 'SKIP'}  GRID:{params['grid_step']}  BLK_R:{params['blk_r']}  SRCH_R:{params['srch_r']}  TEX:{params['min_tex']}  SAD:{params['max_sad']}", col_gate),
    ]

    lh = max(14, int(round(14 * ui_scale)))
    text_y = max(10, int(round(10 * ui_scale)))
    y0 = frame_h - lh * len(lines) - max(4, int(round(4 * ui_scale)))
    for i, (txt, col) in enumerate(lines):
        _txt(vis, txt, max(4, int(round(4 * ui_scale))), y0 + i * lh + text_y,
             color=col, scale=0.36 * ui_scale, thickness=max(1, int(round(ui_scale))))


# ══════════════════════════════════════════════════════════════════════════════
#  GRAYSCALE DOWNSCALE (mirrors firmware pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def bgr_to_gray_downscale(bgr: np.ndarray, ofd_w: int, ofd_h: int) -> np.ndarray:
    """Downscale BGR frame to (ofd_h, ofd_w) grayscale."""
    small = cv2.resize(bgr, (ofd_w, ofd_h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


# ══════════════════════════════════════════════════════════════════════════════
#  TRACKBAR SETUP & READ  (interactive mode only)
# ══════════════════════════════════════════════════════════════════════════════

def _setup_trackbars(params: dict):
    """Create OFD Parameters window with all tunable trackbars."""
    cv2.namedWindow(TB_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TB_WIN, 420, 390)

    def _tb(name, val, maxv):
        cv2.createTrackbar(name, TB_WIN, int(val), int(maxv), lambda _: None)

    _tb("GRID_STEP",      params["grid_step"],                    20)
    _tb("BLK_R",          params["blk_r"],                        12)
    _tb("SRCH_R",         params["srch_r"],                       15)
    _tb("MIN_TEX",        params["min_tex"],                      200)
    _tb("MAX_SAD /100",   params["max_sad"] // 100,               150)
    _tb("ALPHA x100",     int(params["ema_alpha"] * 100),          99)
    _tb("BIAS x1000+100", int(params["div_bias"] * 1000) + 100,   200)
    _tb("DIVTHR x1000",   int(params["div_threshold"] * 1000),    200)
    _tb("TAU_EVADE ms",   int(params["tau_evade"]),                500)
    _tb("TAU_BRAKE ms",   int(params["tau_brake"]),                500)
    _tb("AZ_BAND x100",   int(params["az_band"] * 100),            50)


def _read_trackbars(params: dict) -> bool:
    """Read all trackbars and update params. Returns True if anything changed."""
    gs  = max(2, cv2.getTrackbarPos("GRID_STEP",      TB_WIN))
    br  = max(1, cv2.getTrackbarPos("BLK_R",          TB_WIN))
    sr  = max(2, cv2.getTrackbarPos("SRCH_R",         TB_WIN))
    mt  = max(1, cv2.getTrackbarPos("MIN_TEX",        TB_WIN))
    ms  = max(100, cv2.getTrackbarPos("MAX_SAD /100",  TB_WIN) * 100)
    al  = max(0.01, cv2.getTrackbarPos("ALPHA x100",   TB_WIN) / 100.0)
    bi  = (cv2.getTrackbarPos("BIAS x1000+100", TB_WIN) - 100) / 1000.0
    dt  = max(0.001, cv2.getTrackbarPos("DIVTHR x1000", TB_WIN) / 1000.0)
    te  = max(5.0, float(cv2.getTrackbarPos("TAU_EVADE ms", TB_WIN)))
    tb2 = max(5.0, float(cv2.getTrackbarPos("TAU_BRAKE ms", TB_WIN)))
    ab  = max(0.01, cv2.getTrackbarPos("AZ_BAND x100",  TB_WIN) / 100.0)

    new = dict(grid_step=gs, blk_r=br, srch_r=sr, min_tex=mt, max_sad=ms,
               ema_alpha=al, div_bias=bi, div_threshold=dt,
               tau_evade=te, tau_brake=tb2, az_band=ab)
    changed = any(params.get(k) != v for k, v in new.items())
    params.update(new)
    return changed


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PROCESSING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run(input_path: str, args, params: dict):
    stem     = os.path.splitext(input_path)[0]
    csv_path = resolve_csv_path(input_path)

    print(f"\n{'─'*60}")
    print(f"Input : {input_path}")

    try:
        source = Mp4Source(input_path)
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    frame_count = source.frame_count
    fps = source.fps
    width = source.width
    height = source.height

    print(f"Type  : .mp4 via OpenCV")
    print(f"Frames: {frame_count}  |  {width}×{height}  |  {fps}fps")

    # ── OFD resolution ────────────────────────────────────────────────────────
    ofd_w, ofd_h = args.ofd_res
    print(f"OFD   : {ofd_w}×{ofd_h}  |  scale={args.scale}×")

    # ── CSV sidecar (full IMU + OFD) ─────────────────────────────────────────
    imu, _ = load_ofd(csv_path) if csv_path else ({}, False)
    if imu:
        csv_frames = sorted(int(k) for k in imu.keys())
        csv_first = csv_frames[0]
        csv_last = csv_frames[-1]
        print(f"CSV   : {len(imu)} rows loaded  |  frames {csv_first}..{csv_last}")
        if csv_first != 1:
            print(f"NOTE  : CSV starts at frame {csv_first}; frame sync uses CSV 'frame' numbers directly")
        synced_frame_count = min(frame_count, csv_last)
        if synced_frame_count != frame_count:
            print(f"Sync  : limiting playback to {synced_frame_count} frames to match CSV coverage")
            frame_count = synced_frame_count
    else:
        print("CSV   : not found — IMU panel will show zeros")

    # ── output setup ─────────────────────────────────────────────────────────
    disp_w   = width  * args.scale
    disp_h   = height * args.scale
    play_fps = args.fps if args.fps else fps
    ui_scale = compute_ui_scale(disp_w, disp_h)
    panel_w  = max(PANEL_W, int(round(PANEL_W * ui_scale)))

    show_panel = True
    total_w    = disp_w + panel_w   # full canvas width (panel always shown in save mode)

    writer = None
    if args.save:
        out_path = stem + "_vizflow.mp4"
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc, float(play_fps), (total_w, disp_h))
        if not writer.isOpened():
            print(f"ERROR: cannot open writer → {out_path}"); return
        print(f"Save  : {out_path}")
    else:
        _setup_trackbars(params)
        win = "OFD Flow Visualizer  [Space=step  P=play  S=save  R=reset  T=panel  Q=quit]"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, total_w, disp_h)

    opts = dict(show_search=args.show_search,
                show_block=args.show_block,
                show_div=args.show_div,
                show_tex=args.show_tex)

    ofd_filt = OfdFilter(params)
    playing   = args.save

    frame_delay_ms = max(1, int(1000.0 / play_fps))

    # Gray frame cache — lets us seek backwards without re-reading disk.
    # 320×240 grayscale ≈ 75 KB/frame; 1000 frames ≈ 73 MB (acceptable).
    gray_cache: dict[int, np.ndarray] = {}

    try:
        frame_num = 1
        while 1 <= frame_num <= frame_count:

            # ── read frame from source (supports seek for interactive stepping) ──
            bgr = source.read_frame(frame_num)
            if bgr is None:
                print(f"\nWARNING: truncated at frame {frame_num}")
                break

            gray = bgr_to_gray_downscale(bgr, ofd_w, ofd_h)
            gray_cache[frame_num] = gray          # cache for back-navigation

            # ── read trackbars & rebuild filter if params changed ─────────────
            if not args.save:
                if _read_trackbars(params):
                    ofd_filt = OfdFilter(params)

            # ── full IMU row from CSV ─────────────────────────────────────────
            row = imu.get(frame_num, {})
            az    = float(row.get("az",    params["az_center"]))
            ts    = float(row.get("timestamp_ms", frame_num * (1000.0 / fps)))
            imu_row = dict(
                ax    = float(row.get("ax",    0.0)),
                ay    = float(row.get("ay",    0.0)),
                az    = az,
                gx    = float(row.get("gx",   0.0)),
                gy    = float(row.get("gy",   0.0)),
                gz    = float(row.get("gz",   0.0)),
                roll  = float(row.get("roll",  0.0)),
                pitch = float(row.get("pitch", 0.0)),
                yaw   = float(row.get("yaw",   0.0)),
            )

            # ── OFD ──────────────────────────────────────────────────────────
            raw_ofd = dict(divergence=0.0, tau=0.0, vx_mean=0.0, vy_mean=0.0,
                           lr_balance=0.0, flow_cnt=0, div_cnt=0, valid=0)
            points  = []

            prev_gray = gray_cache.get(frame_num - 1)
            if prev_gray is not None:
                raw_ofd, points = ofd_process_gray_viz(gray, prev_gray, params)

            filt = ofd_filt.step(raw_ofd, ts, az)
            raw_disp, filt_disp = build_csv_display_state(row, raw_ofd, filt, params)

            # ── build display canvas ──────────────────────────────────────────
            cur_panel = show_panel or args.save
            canvas_w  = disp_w + (panel_w if cur_panel else 0)
            canvas    = np.zeros((disp_h, canvas_w, 3), np.uint8)

            vis = cv2.resize(bgr, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
            draw_flow_overlay(vis, points, params,
                              scale=args.scale,
                              ofd_w=ofd_w, ofd_h=ofd_h,
                              frame_w=disp_w, frame_h=disp_h,
                              opts=opts)
            draw_stats_overlay(vis, raw_disp, filt_disp, frame_num, params, disp_h, ui_scale=ui_scale)
            canvas[:, :disp_w] = vis

            if cur_panel:
                panel_slice = canvas[:, disp_w:]
                draw_decision_panel(panel_slice, filt_disp, imu_row, params, ui_scale=ui_scale)

            # ── output ────────────────────────────────────────────────────────
            if args.save:
                writer.write(canvas)
                if frame_num % 30 == 0 or frame_num == frame_count:
                    pct = frame_num / frame_count * 100
                    bar = ("█" * (frame_num * 30 // frame_count)).ljust(30)
                    print(f"  [{bar}] {frame_num}/{frame_count} ({pct:.0f}%)", end="\r")
                frame_num += 1
            else:
                cv2.imshow(win, canvas)
                delay  = frame_delay_ms if playing else 0
                raw_key  = cv2.waitKeyEx(delay)   # waitKeyEx returns full code for arrow keys
                key      = raw_key & 0xFF if raw_key != -1 else 255

                # Arrow keys — waitKeyEx values:
                #   Windows: Left=2424832  Right=2555904
                #   Linux:   Left=65361    Right=65363
                is_left  = raw_key in (2424832, 65361)   # ←
                is_right = raw_key in (2555904, 65363)   # →

                if key in (ord("q"), 27):           # Q / Esc → quit
                    break
                elif key == ord("p"):               # P → toggle play
                    playing = not playing
                elif key == ord("r"):               # R → reset filter
                    ofd_filt = OfdFilter(params)
                    print(f"\nFilter reset at frame {frame_num}")
                elif key == ord("t"):               # T → toggle panel
                    show_panel = not show_panel
                    new_w = disp_w + (panel_w if show_panel else 0)
                    cv2.resizeWindow(win, new_w, disp_h)
                elif key == ord("s"):               # S → save PNG
                    png = f"{stem}_F{frame_num:04d}.png"
                    cv2.imwrite(png, canvas)
                    print(f"\nSaved: {png}")

                # ── navigation ───────────────────────────────────────────────
                if is_left:
                    # Drain queued key-repeats so holding skips multiple frames
                    skip = 1
                    while cv2.waitKeyEx(1) == raw_key:
                        skip += 1
                        if skip >= 60:
                            break
                    frame_num = max(1, frame_num - skip)
                    playing = False
                    ofd_filt = OfdFilter(params)   # reset EMA; history is invalid going back
                elif is_right:
                    skip = 1
                    while cv2.waitKeyEx(1) == raw_key:
                        skip += 1
                        if skip >= 60:
                            break
                    frame_num = min(frame_count, frame_num + skip)
                elif playing or key == ord(" "):
                    frame_num += 1
                # else: paused + non-advance key → redisplay same frame
    finally:
        source.close()

    if writer:
        writer.release()
        print(f"\nDone  : {frame_count} frames → {out_path}")
    else:
        cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_res(s: str):
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid resolution '{s}' — expected WxH e.g. 320x240")


def main():
    ap = argparse.ArgumentParser(
        description="Optical flow vector visualizer for OFD parameter tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("input", help="Input .mp4 to visualize")

    # ── block-matching params ─────────────────────────────────────────────────
    g = ap.add_argument_group("Block-matching parameters")
    g.add_argument("--grid-step", type=int,   default=DEFAULTS["grid_step"],
                   metavar="N", help=f"Grid spacing in pixels (default {DEFAULTS['grid_step']})")
    g.add_argument("--blk-r",    type=int,   default=DEFAULTS["blk_r"],
                   metavar="N", help=f"Block radius → (2R+1)² block (default {DEFAULTS['blk_r']})")
    g.add_argument("--srch-r",   type=int,   default=DEFAULTS["srch_r"],
                   metavar="N", help=f"Search radius in prev frame (default {DEFAULTS['srch_r']})")
    g.add_argument("--min-tex",  type=int,   default=DEFAULTS["min_tex"],
                   metavar="N", help=f"Min texture threshold (default {DEFAULTS['min_tex']})")
    g.add_argument("--max-sad",  type=int,   default=DEFAULTS["max_sad"],
                   metavar="N", help=f"Max SAD to accept match (default {DEFAULTS['max_sad']})")

    # ── filter params ─────────────────────────────────────────────────────────
    ff = ap.add_argument_group("Filter parameters")
    ff.add_argument("--alpha",      type=float, default=DEFAULTS["ema_alpha"],
                   metavar="F", help=f"EMA alpha (default {DEFAULTS['ema_alpha']})")
    ff.add_argument("--div-bias",   type=float, default=DEFAULTS["div_bias"],
                   metavar="F", help=f"Divergence DC bias (default {DEFAULTS['div_bias']})")
    ff.add_argument("--tau-evade",  type=float, default=DEFAULTS["tau_evade"],
                   metavar="MS", help=f"Evasion τ threshold ms (default {DEFAULTS['tau_evade']})")
    ff.add_argument("--tau-brake",  type=float, default=DEFAULTS["tau_brake"],
                   metavar="MS", help=f"Brake τ threshold ms (default {DEFAULTS['tau_brake']})")

    # ── display params ────────────────────────────────────────────────────────
    d = ap.add_argument_group("Display")
    d.add_argument("--ofd-res",    type=_parse_res, default="160x120",
                   metavar="WxH", help="OFD processing resolution (default 160x120)")
    d.add_argument("--scale",      type=int, default=2,
                   metavar="N",   help="Display scale factor (default 2)")
    d.add_argument("--fps",        type=int, default=None,
                   metavar="N",   help="Playback FPS override (default: from MP4 metadata)")
    d.add_argument("--save",       action="store_true",
                   help="Export MP4 instead of interactive display")
    d.add_argument("--show-search", action="store_true",
                   help="Draw search radius circles")
    d.add_argument("--show-block",  action="store_true",
                   help="Draw block-size boxes at each grid point")
    d.add_argument("--show-div",    action="store_true",
                   help="Per-cell divergence heatmap tint")
    d.add_argument("--show-tex",    action="store_true",
                   help="Texture score heatmap tint")

    args = ap.parse_args()

    if not os.path.isfile(args.input):
        ap.error(f"File not found: {args.input}")
    if pathlib.Path(args.input).suffix.lower() != ".mp4":
        ap.error("Only .mp4 input is supported. Convert .MJP to .mp4 first.")

    params = dict(DEFAULTS)
    params.update(
        grid_step  = args.grid_step,
        blk_r      = args.blk_r,
        srch_r     = args.srch_r,
        min_tex    = args.min_tex,
        max_sad    = args.max_sad,
        ema_alpha  = args.alpha,
        div_bias   = args.div_bias,
        tau_evade  = args.tau_evade,
        tau_brake  = args.tau_brake,
    )

    run(args.input, args, params)


if __name__ == "__main__":
    main()
