# CLAUDE.md — FWR Vision Firmware
## Flapping-Wing Robot | ESP32-P4 | Active Research Prototype

> **THIS IS AN ACTIVE RESEARCH PROTOTYPE.**
> Do not assume any subsystem is production-ready or complete.
> This document reflects the actual verified code state as of 2026-05-10.

---

## Project Overview

| Property | Value |
|---|---|
| Target MCU | ESP32-P4 (dual-core Xtensa LX7, PSRAM, hardware JPEG) |
| IDF version | ESP-IDF v5.5.1 |
| CMake minimum | 3.16 |
| Build system | ESP-IDF component model (idf.py / ninja) |
| Camera | OV5647 (5MP, 62° HFOV, I²C SCCB) |
| IMU | WITmotion WT901B (I²C, 0x50) |
| RC input | SBUS via UART (100kbps, inverted, 8N2) |
| Storage | microSD via SDMMC 4-bit HS (~25 MB/s) |

### Actual Current Capability
- Manual RC flapping-wing flight with servo control
- SD video recording (MJPEG at 1280×960, ~30 fps)
- OFD (optical flow divergence) computed and logged per frame
- IMU data logged alongside OFD in sidecar CSV
- WiFi MJPEG streaming: **NOT present** (referenced in prior notes but not found in source)
- Autonomous obstacle avoidance: detection pipeline complete, servo actuation **commented out**

### Intended Capability
Full closed-loop obstacle avoidance: detect looming → compute evasion → inject servo offset → turn away from obstacle autonomously during flight.

### Current Maturity Level
- **Detection/logging:** solid prototype — tuned on real flight data (V0000.CSV, 1536 frames)
- **Autonomous avoidance:** one integration step behind — all components exist, injection not enabled

---

## Repository Reality Check

| Subsystem | Status | Notes |
|---|---|---|
| `main_fwr_ofd.cpp` | **Functional** | Active entrypoint, runs every boot |
| `ofd.cpp` / `ofd_config.h` | **Functional** | Tuned on real data, gyro-derotated flow |
| `video_rec.cpp` | **Functional** | MJPEG + CSV recording pipeline |
| `fwr_control.cpp` | **Functional** | Servo PWM + sin-wave wing stroke |
| `sbus_rx.cpp` | **Functional** | SBUS decode, failsafe, 8-channel |
| `fwr_set_ofd_avoidance()` | **Partial** | API exists in fwr_control; caller is commented out |
| `yolo.cpp` | **Compiled, unused** | Embedded as SRCS but never called from active main |
| `main.cpp` | **Broken / deprecated** | Calls old 1-arg `ofd_process_gray()` — compile error if re-enabled |
| `main_fwr_test.cpp` | **Disabled** | Commented out in CMakeLists; purpose unclear |
| `main_imu_test.cpp` | **Disabled** | Standalone IMU probe; useful for hardware debug |
| `main_sbus_monitor.cpp` | **Disabled** | SBUS channel monitor; useful for radio setup |
| EKF (`ofd_ekf.cpp`) | **Does not exist** | Not found anywhere in repo; prior reference was incorrect |
| `lcd.c` / `lcd.h` | **Unused in active build** | LCD driver compiled as component; not called from active main |
| WiFi streaming | **Not present** | No WiFi/MJPEG server code found in any file |
| `video converter/*.py` | **Functional tools** | Post-processing scripts: convert, retune, visualize |

---

## Active Firmware Path

### Entrypoint Selection (CMakeLists.txt)

```cmake
# main/CMakeLists.txt — only ONE main is active:
"main_fwr_ofd.cpp"     ← ACTIVE

# Commented out (inactive):
# "main.cpp"
# "main_sbus_monitor.cpp"
# "main_imu_test.cpp"
# "main_fwr_test.cpp"
```

### Boot Initialization Order

```
app_main() [main_fwr_ofd.cpp]
 │
 ├─ fwr_control_init()       → LEDC PWM timer init, servos → 1500µs neutral
 ├─ fwr_control_start()      → Launch sin_timer (3ms) + servo_timer (4ms)
 ├─ sbus_rx_init()           → UART1 config (100kbps, 8E2, inverted RXD)
 ├─ sbus_rx_start()          → Launch sbus_task (Core 1, priority 10)
 ├─ video_rec_init()
 │   ├─ sdcard_init()        → Mount FAT32, benchmark, cluster check
 │   ├─ Alloc DMA bounce buf → 64KB internal SRAM
 │   ├─ Alloc PSRAM ring buf → 1MB PSRAM (SPSC ring for SD streaming)
 │   ├─ Alloc CSV buf        → 128KB PSRAM
 │   ├─ Create frame queues  → write_queue + free_queue (3 slots each)
 │   ├─ Scan SD for V*.MJP   → set next video_id
 │   ├─ ofd_init(160, 120)   → Alloc prev-frame buffer, set grid params
 │   └─ JPEG encoder init    → HW encoder, 1MB output buffer
 ├─ imu_init()               → I2C_NUM_1 bus, probe 0x50/0x51, wait up to 5s
 ├─ camera_set_sensor(OV5647)
 ├─ camera_set_resolution(1280, 960)
 └─ camera_init()            → V4L2 /dev/video0, 6 MMAP buffers, RGB565
```

**Main loop (after boot):**
```
while(1):
  1. Check SW1 (RC CH4): UP→start_recording(), DOWN→stop_recording()
  2. imu_read() → video_rec_set_imu()
  3. camera_get_frame() [blocks on V4L2 DQBUF]
  4. video_rec_enqueue() → hands frame to writer_task
  5. video_rec_last_ofd() → read latest OFD result
  6. [COMMENTED OUT] fwr_set_ofd_avoidance(ail, 0)
  7. vTaskDelay(5ms) only if not recording
```

---

## Full System Architecture

### Task Map

| Task | Core | Priority | Stack | Role |
|---|---|---|---|---|
| `app_main` (main loop) | 0 | — | Default | Camera capture, IMU read, SW1 control |
| `sbus_task` | 1 | **10** | 4096 | SBUS decode, writes `aileron`/`elevator`/`throttle`/`rudder`/`frequency` |
| `writer_task` | 1 | 5 | 8192 | JPEG encode, OFD dispatch, CSV append, ring push |
| `sd_stream_task` | 1 | 3 | 4096 | Drains SPSC ring → SD in 64KB chunks |
| `ofd_task` | 0 | 3 | 4096 | Block matching, gyro derotation, EMA filter, looming trigger |
| `sin_timer` (esp_timer) | — | — | — | 3ms sinusoid for wing stroke |
| `servo_timer` (esp_timer) | — | — | — | 4ms servo PWM update |

> Note: `ofd_task` and `sd_stream_task` only exist during an active recording session. They are created at `start_recording()` and destroyed at `stop_recording()`.

### Hardware Bus Map

| Bus | GPIO | Devices | Notes |
|---|---|---|---|
| I2C_NUM_0 | SCL=GPIO7, SDA=GPIO8 | OV5647 SCCB | Camera sensor config |
| I2C_NUM_1 | SCL=GPIO24, SDA=GPIO25 | WT901B IMU | Independent from camera |
| UART1 | RX=GPIO20, TX=GPIO18 | SBUS receiver | 100kbps, inverted, 8E2 |
| SDMMC 4-bit | CLK=43, CMD=44, D0-D3=39-42 | microSD | HS mode ~25 MB/s |
| LEDC CH0 | GPIO4 | Left servo | 250Hz, 16-bit |
| LEDC CH1 | GPIO5 | Right servo | 250Hz, 16-bit |

---

## Optical Flow Divergence Pipeline

### Stage 1 — Frame Acquisition (writer_task, Core 1)
```
RGB565 frame (1280×960) from camera MMAP
  ↓ extract_luma_from_rgb565()
Grayscale 160×120 (nearest-neighbor downsample + (77R+150G+29B)>>8 luma)
  ↓ pushed via ofd_queue to ofd_task
```

### Stage 2 — Block Matching (ofd_task, Core 0)
```
Parameters: GRID_STEP=12, BLK_R=4 (9×9 block), SRCH_R=6, MIN_TEX=25, MAX_SAD=4500
Grid spans: (margin=12) to (160-13) horizontally, (12) to (120-13) vertically
For each grid point:
  1. texture_block() — sum |gradients| in 9×9 neighborhood
     → skip if < MIN_TEX (low texture → unreliable match)
  2. sad_block() exhaustive search over (2×SRCH_R+1)² = 169 candidates
     → raw flow: u_raw=-bestDx, v_raw=-bestDy
     → reject if bestSAD > 4500
  3. Gyro derotation:
     wx=-gx·π/180, wy=-gy·π/180 (X/Y flip: camera roll≈-180°), wz=+gz·π/180
     xn=(x-80)/130, yn=(y-60)/130  (fx=fy=130px at 160×120)
     u_rot = fx·(xn·yn·wx - (1+xn²)·wy + yn·wz)
     v_rot = fy·((1+yn²)·wx - xn·yn·wy - xn·wz)
     u_derot = u_raw - u_rot
     v_derot = v_raw - v_rot
  4. Divergence estimation (finite differences on derotated grid):
     du/dx via right neighbor: (u[c+1] - u[c]) / GRID_STEP
     dv/dy via above-row:     (v_cur[c] - v_prev[c]) / GRID_STEP
     div_here = du/dx + dv/dy  (per-cell)
  5. L/R split: left half columns → div_left_sum, right → div_right_sum
  6. Accumulate: div_sum, flow_mag_sum, flow_cnt, div_cnt
```

### Stage 3 — Filter Layer (ofd_task, still)
```
Requires: div_cnt ≥ OFD_MIN_DIV_CNT (10)

Bias subtraction: corrected = divergence - OFD_DIV_BIAS (-0.018)
EMA (α=0.30):
  ema_div      = 0.30·corrected + 0.70·ema_div
  ema_lr       = 0.30·lr_balance + 0.70·ema_lr
  ema_flow_mag = 0.30·mean_flow_mag + 0.70·ema_flow_mag

τ (time-to-contact, logging only):
  dt clamped to [8ms, 80ms]  ← kills 1106ms spike artifacts
  tau_ms = ema_div / (Δema_div/dt)  — NOT used for decisions
```

### Stage 4 — Trigger Logic
```
PRIMARY gate:  ema_flow_mag > OFD_FLOW_THRESH_BRAKE (3.5 px/frame)
SECONDARY gate: ema_flow_mag > OFD_FLOW_THRESH_ALERT (2.0) AND |ema_div| > OFD_DIV_THRESHOLD (0.05)

flow_trigger = PRIMARY OR SECONDARY
looming = flow_trigger AND (div_cnt ≥ 10) AND valid

evasion_level:
  NONE  (0): looming = false
  ALERT (1): ema_flow_mag > 2.0
  BRAKE (2): ema_flow_mag > 3.5   ← current primary trigger
  EVADE (3): ema_flow_mag > 5.0

turn_cmd = clamp(-ema_lr × OFD_LR_GAIN, -1, 1)   (OFD_LR_GAIN=3.0)
```

### Stage 5 — Wing-Sync Gate (optional, currently DISABLED)
```
OFD_USE_AZ_QUIET_GATE = 0  → gate always passes (true)
When enabled: skip OFD frame if az ∉ [-0.986 ± 0.15] g
Purpose: exclude wing-stroke-vibration frames from flow computation
```

### Stage 6 — Output Storage
```
store_last_ofd(r)  [protected by portMUX_TYPE s_ofd_lock critical section]
  → accessible via video_rec_last_ofd() from main loop
  → logged to CSV by writer_task (reads s_last_ofd via load_last_ofd())
```

---

## Obstacle Avoidance State

### Where the Pipeline Currently Stops

```
ofd_task computes: looming_detected, evasion_level, turn_cmd ✓
main loop reads via video_rec_last_ofd() ✓
main loop computes ail = (int)(r.turn_cmd * MAX_AVOID_US)  [MAX_AVOID_US=600] ✓
main loop calls (void)ail  ← DISCARDED
fwr_set_ofd_avoidance(ail, 0) ← COMMENTED OUT  (line 228, main_fwr_ofd.cpp)
```

### To Enable Avoidance
Uncomment two lines in `main_fwr_ofd.cpp:228`:
```cpp
// BEFORE:
// fwr_set_ofd_avoidance(ail, 0);   /* TODO: uncomment to enable avoidance */
// ...
// fwr_set_ofd_avoidance(0, 0);

// AFTER (minimal change):
fwr_set_ofd_avoidance(ail, 0);
// ...
fwr_set_ofd_avoidance(0, 0);
```

`fwr_set_ofd_avoidance()` in `fwr_control.cpp:196` sets `ofd_aileron_offset` and `ofd_rudder_offset` (volatile ints), which are already mixed into servo output by `servo_timer_cb()`:
```cpp
int ail = aileron + ofd_aileron_offset;  // ← already there
int rud = rudder  + ofd_rudder_offset;
```

**The servo mixing infrastructure is fully complete.** Only the call is missing.

### Avoidance Decision Chain
```
Detection:  ofd_task → flow_trigger + looming_detected           ✓ working
Decision:   ofd_task → evasion_level, turn_cmd computed          ✓ working
Control:    main loop → ail = turn_cmd × 600µs                   ✓ computed
Actuation:  fwr_set_ofd_avoidance() → ofd_aileron_offset         ✗ commented out
Mixing:     servo_timer_cb → ail = aileron + ofd_aileron_offset  ✓ works
```

### Known Logic Issue (from Prior Review — VERIFIED)
The `evasion_level` suppression concern is **not a real problem**. Inspecting `ofd_task`:
- `looming` and `evasion_level` are set in the same block; `evasion_level` is NONE when `looming=false`
- No independent path sets `evasion_level` without also setting `looming_detected`
- However, main loop reads `r.looming_detected` directly — using `r.evasion_level` for graduated response is cleaner and recommended for the next integration step

---

## Timing / Synchronization

### Frame Timing
```
camera_get_frame() [blocks] → ~16ms at 60fps (OV5647 @ 1280×960)
video_rec_enqueue() → non-blocking queue send (drops if write_queue full)
writer_task JPEG encode → ~20ms (hardware accelerated)
ofd_task block matching → ~5-15ms per frame at 160×120
```

### Known Race Conditions

**RACE 1 — `s_last_imu` unprotected (CONFIRMED BUG)**
- Written by: main loop (`video_rec_set_imu()` → `s_last_imu = d` — simple assignment)
- Read by: `writer_task` (`imu = s_last_imu` in CSV append, line ~847)
- `imu_data_t` is a 9-float struct (36 bytes) — not atomically readable on any architecture
- Fix: add `portMUX_TYPE s_imu_lock` critical section around reads/writes, same pattern as `s_ofd_lock`

**RACE 2 — `sw1_raw` written by sbus_task, read by main loop**
- `volatile int sw1_raw` — single int; acceptable on 32-bit MCU with aligned access
- Technically UB in C++ but practically safe for single-word read/write on ESP32

**RACE 3 — CSV logging coherence (CONFIRMED STRUCTURAL ISSUE)**
- writer_task logs `s_last_ofd` (race-protected) and `s_last_imu` (NOT race-protected) per frame
- The ofd result may be from a **different frame** than the camera frame being logged
- The IMU sample may be from **any point** during that frame interval
- Frame N in the CSV does not necessarily reflect the OFD state computed for frame N
- This is acceptable for prototype-stage post-processing but misleading for flight dynamics analysis

**RACE 4 — Ring buffer SPSC relies on volatile without memory barriers**
- `s_ring_wr` and `s_ring_rd` are `volatile size_t` (not `_Atomic`)
- On dual-core Xtensa, this may require explicit memory barriers for correctness
- Practically works due to FreeRTOS task scheduling, but is technically undefined behavior

### Async Processing Model
```
Core 0: app_main loop
  → camera_get_frame() (blocking)
  → video_rec_enqueue() (non-blocking)
  → video_rec_last_ofd() reads stale OFD (from previous frame batch)
  → OFD result latency: 1-3 frames behind camera

Core 0 also: ofd_task (priority 3)
  → preempted by main loop camera capture? Yes — camera blocks in DQBUF,
    during which ofd_task runs. This is actually well-designed.

Core 1: writer_task (priority 5) + sd_stream_task (priority 3) + sbus_task (priority 10)
  → sbus_task always preempts writer/SD when new bytes arrive
  → writer_task preempts SD task during JPEG encode (~20ms)
  → SD task drains during writer idle time
```

---

## Control Integration

### Servo Mixing Formula (servo_timer_cb, 4ms period)
```
left  = 1500 + (-elevator + aileron + ofd_aileron_offset + servo_adj_l)
             + (throttle + rudder + ofd_rudder_offset) × sin(ωt)

right = 1500 + ( elevator + aileron + ofd_aileron_offset + servo_adj_r)
             + (-throttle + rudder + ofd_rudder_offset) × sin(ωt)
```
Range clamped to [900, 2100] µs.

### RC Channel Mapping

| CH | Variable | Range (µs offset) | Control |
|---|---|---|---|
| 0 | aileron | ±300 | Roll / bank |
| 1 | elevator | ±400 | Pitch |
| 2 | throttle | 0..+600 | Wing amplitude |
| 3 | rudder | ±300 | Yaw |
| 4 | sw1_raw | raw SBUS | Recording switch (SW1) |
| 5 | frequency | 1000→0 (inverted) | Wing stroke frequency |
| 6 | servo_adj_l | ±100 | Left servo trim |
| 7 | servo_adj_r | ±100 | Right servo trim |

### OFD Avoidance Injection (DISABLED)
```cpp
// What it does when enabled:
ofd_aileron_offset = ail;  // turn_cmd × 600µs, range [-600, +600]
ofd_rudder_offset  = 0;    // rudder injection disabled (USE_ROLL=1)
```
Impact: OFD overrides ≤600µs on aileron, on top of RC aileron command. No RC override lockout — OFD and pilot commands simply add.

### SBUS Failsafe
If no SBUS frame received for 100ms: `throttle=0, frequency=0` (wing stops). Aileron/elevator/rudder retain last values — intentional for glide behavior.

---

## Experimental / Inactive Code

### `main.cpp` — DEPRECATED, BROKEN
- References `crop_to_lcd()` which is in `lcd.h` (not included via CMakeLists when active)
- Calls `ofd_process_gray(ofd_gray)` — **1-argument call**, but current signature requires 4 arguments (`const uint8_t*, float, float, float`)
- **Would fail to compile** if uncommented in CMakeLists.txt
- Uses camera at 800×640 not 1280×960
- All YOLO calls commented out within the file itself
- Has primitive OFD-based avoidance logic (no EMA, raw divergence threshold 0.08)

### `yolo.cpp` — COMPILED, NEVER CALLED
- Compiled into the build via CMakeLists SRCS
- `yolo_init()` and `yolo_run()` are never called from `main_fwr_ofd.cpp`
- Embeds `bas.espdl` (160×160, basketball detection model) — 100+ KB flash cost with no benefit currently
- Could be removed from SRCS to reduce flash usage, or integrated for object-level avoidance

### `lcd.c` / `lcd.h` — COMPONENT PRESENT, UNUSED IN FLIGHT
- LCD component is a CMake REQUIRES dependency
- No `lcd_init()` or `lcd_draw()` call in `main_fwr_ofd.cpp`
- Was used in old `main.cpp` for live display

### EKF (`ofd_ekf.cpp/.h`) — DOES NOT EXIST
- Prior documentation claimed these files exist but are untracked
- **Verified: no EKF files found anywhere in the repository**
- This was an incorrect claim in the prior review

### `model/*.espdl` — MULTIPLE UNUSED MODELS
```
model/bas.espdl         ← ACTIVE (embedded by CMakeLists)
model/bas2.espdl        ← unused variant
model/bas224.espdl      ← unused variant
model/basketball_64.espdl   ← unused, commented out
model/basketball_640.espdl  ← unused, commented out
model/ESPdet.espdl      ← unused, commented out
model/kyu160.espdl      ← unused
model/kyubest1.espdl    ← unused, commented out
model/kyumini.espdl     ← unused
```

---

## Folder Structure

```
FWR_vision/
├── CMakeLists.txt              Root ESP-IDF project (project name: FWR_VISION)
├── sdkconfig                   ESP-IDF menuconfig (DO NOT edit manually)
├── sdkconfig.defaults          Committed defaults: esp32p4, QIO 16MB, -O3, WDT off
├── partitions.csv              NVS 24KB + PHY 4KB + factory 7MB
├── dependencies.lock           Managed component versions (do not edit)
├── .clangd                     Clang language server config
│
├── main/
│   ├── CMakeLists.txt          ← Controls which main is active; MODIFY HERE to switch
│   ├── Kconfig                 Menuconfig options (currently minimal)
│   ├── idf_component.yml       Managed component declarations
│   │
│   ├── main_fwr_ofd.cpp        ★ ACTIVE ENTRYPOINT — manual flight + OFD + SD recording
│   ├── main.cpp                ✗ DEPRECATED — broken API, YOLO+LCD+OFD prototype
│   ├── main_fwr_test.cpp       ✗ DISABLED — servo/SBUS test (not reviewed)
│   ├── main_imu_test.cpp       ✗ DISABLED — standalone IMU probe, useful for hw debug
│   ├── main_sbus_monitor.cpp   ✗ DISABLED — SBUS channel printout, useful for radio setup
│   │
│   ├── ofd.cpp                 ★ OFD core: block matching, gyro derotation, divergence
│   ├── ofd.h                   OFD public API + ofd_result_t struct definition
│   ├── ofd_config.h            ★ All OFD tuning constants — PRIMARY TUNING FILE
│   ├── video_rec.cpp           ★ Recording pipeline: SD, JPEG, OFD task, CSV logging
│   ├── video_rec.h             Recording API: init/start/stop/enqueue/last_ofd
│   ├── yolo.cpp                YOLO ESP-DL inference (compiled but unused in active main)
│   ├── yolo.h                  YOLO API
│   │
│   ├── fwr_control/
│   │   ├── fwr_control.cpp     ★ Servo PWM, wing sin generator, OFD avoidance injection
│   │   ├── fwr_control.h       Servo API: init/start/set_ofd_avoidance
│   │   ├── sbus_rx.cpp         SBUS UART decode, failsafe, channel mapping
│   │   └── sbus_rx.h           SBUS API: init/start
│   │
│   └── model/                  ESP-DL model binaries (various YOLO variants)
│       └── bas.espdl           ← Only one embedded in build
│
├── components/
│   ├── camera/
│   │   ├── camera.c            V4L2 camera interface, OV5647/SC2336 selection
│   │   └── camera.h            Camera API: set_sensor/set_resolution/init/get_frame/return_frame
│   ├── lcd/                    LCD driver (EK79007/HX8394) — unused in active firmware
│   └── example_video_common/   ISP/JPEG HW encoder helpers (esp-idf example port)
│       ├── example_encoder.c   Hardware JPEG encoder init + process
│       └── example_init_video.c esp_video driver init
│
├── video\ converter/           ★ Post-processing Python tools (run on host PC)
│   ├── convert_mjp.py          Convert V*.MJP to standard MP4
│   ├── convert_vid.py          Convert older .VID format + OFD overlay
│   ├── retune_ofd.py           ★ Re-run OFD pipeline offline with different params
│   ├── viz_ofd.py              Visualize OFD CSV data
│   └── real_flight.*           Sample flight data (CSV + VID + MP4)
│
└── .claude/
    ├── settings.local.json     Allowed bash commands / permissions (project-local)
    └── agents/
        ├── esp32-reviewer.md   ★ Haiku-4-5 agent: RTOS/ISR/ring-buffer reviewer
        └── codebase-explorer.md ★ Haiku-4-5 agent: file/symbol search
```

---

## Build System

### Active Sources (CMakeLists SRCS)
```cmake
ofd.cpp
yolo.cpp                    # compiled but not called — consider removing
fwr_control/fwr_control.cpp
fwr_control/sbus_rx.cpp
video_rec.cpp
main_fwr_ofd.cpp            # THE active main
```

### Compile Options
```cmake
target_compile_options(${COMPONENT_LIB} PRIVATE -O3 -ffast-math)
```
> **Warning:** `-ffast-math` allows FP reassociation and disables NaN/Inf checks. The OFD math uses `fabsf`, `sqrtf`, and `exp` chains — results may differ slightly from IEEE 754. This is intentional for speed on embedded.

### REQUIRES Dependencies
```
esp_driver_i2c, esp_video, esp_cam_sensor, esp_sccb_intf, esp_driver_cam
esp_driver_uart, esp_lcd, esp_lcd_ek79007, driver, esp_timer
unity (test framework — linked but unused in active main)
example_video_common, lcd, camera (custom components)
fatfs, vfs, sdmmc, esp_h264
esp-dl (YOLO inference — flash cost even when not called)
```

### sdkconfig.defaults Key Settings
```
CONFIG_IDF_TARGET=esp32p4
CONFIG_ESPTOOLPY_FLASHMODE_QIO=y         # Quad SPI flash
CONFIG_ESPTOOLPY_FLASHSIZE_16MB=y        # 16MB flash
CONFIG_COMPILER_OPTIMIZATION_PERF=y      # -O2 for IDF components
CONFIG_ESP_TASK_WDT_EN=n                 # WDT DISABLED (dangerous in prod)
CONFIG_SPIRAM_USE_MALLOC=y               # PSRAM as default heap
CONFIG_SPIRAM_MALLOC_ALWAYSINTERNAL=0    # Prefer PSRAM for large allocs
```

> **Danger:** `CONFIG_ESP_TASK_WDT_EN=n` means no watchdog. A deadlock or infinite loop will freeze the MCU silently. This is acceptable for lab development but must be re-enabled before any unsupervised flight.

### Build Command
```bash
# Windows (IDF v5.5.1 — use the correct Python env):
powershell.exe -Command "Set-Location 'C:\Users\thana\01_Fang\Research\04_Firmware_ESP32P4\FWR_vision\build'; & 'C:\Users\thana\.espressif\tools\ninja\1.12.1\ninja.exe'"
```

---

## Hardware Architecture

### ESP32-P4
- Dual-core Xtensa LX7 @ 400MHz
- 768KB internal SRAM (split: instruction cache + DMA-capable data)
- 16MB external PSRAM (accessible via `MALLOC_CAP_SPIRAM`)
- Hardware JPEG encoder (used for 1280×960 @ ~20ms/frame)
- Hardware H.264 encoder (present but unused)
- SDMMC host controller (SDMMC4 4-bit @ 25 MB/s)

### OV5647 Camera
- 5MP CMOS, 62° HFOV (horizontal)
- Connected via V4L2 driver (`/dev/video0`)
- 6 MMAP buffers for zero-copy streaming
- Output at 1280×960: RGB565 (2 bytes/pixel = 2.4 MB/frame raw)
- Alternative sensor SC2336 defined in camera.c but not selected

### WT901B IMU
- 6-axis: 3-axis accelerometer (16g range) + 3-axis gyroscope (2000°/s)
- Also provides fused angles (roll/pitch/yaw)
- I²C address: 0x50 (default), fallback 0x51
- Boot time: ~3 seconds — firmware waits up to 5s
- **Known issue:** Does not support I²C burst register reads reliably; firmware uses 9 individual register reads per sample
- Registers: Accel 0x34-0x36, Gyro 0x37-0x39, Angles 0x3D-0x3F
- Scaling: `raw / 32768 × full_scale` (16g / 2000°/s / 180°)

### Servos
- Two standard RC servos: left (GPIO4) and right (GPIO5)
- PWM: 250Hz, 16-bit resolution, range 900–2100µs, neutral 1500µs
- Wing stroke: sinusoidal modulation at 0–10Hz (tunable via RC CH5)

### SBUS Receiver
- RC receiver output: SBUS protocol (Futaba/FrSky standard)
- 100kbps UART, Even parity, 2 stop bits, logic-inverted
- Hardware inversion via `uart_set_line_inverse(UART_SIGNAL_RXD_INV)`
- 25-byte frames, 11-bit channel encoding, 8 channels decoded

### SD Card
- SDMMC 4-bit wide bus, HS mode (25 MB/s)
- On-chip LDO channel 4 @ 3.3V (no UHS-I to avoid voltage-switch failures)
- GPIOs: CLK=43, CMD=44, D0=39, D1=40, D2=41, D3=42
- FAT32 file system via ESP-IDF VFS (`/sdcard`)
- Cluster warning if < 64KB allocation units

---

## Runtime Mental Model

```
[WT901B IMU] ──I2C_NUM_1──→ imu_read() [main loop, ~400kHz]
                              │
                              ▼
                         s_last_imu  ←────── [RACE: no lock]

[OV5647 Camera] ──V4L2──→ camera_get_frame() [blocks ~16ms]
                              │
                              ▼
                     video_rec_enqueue()
                              │
                     [frame_slot: pointer to MMAP buffer]
                              │
                    write_queue (3 slots)
                              │
                              ▼
                       writer_task [Core 1, pri 5]
                       ├─ JPEG encode (HW, ~20ms)
                       │   → ring_write() → SPSC ring (1MB PSRAM)
                       │        │
                       │        ▼
                       │   sd_stream_task [Core 1, pri 3]
                       │   → fwrite() 64KB chunks → V*.MJP
                       │
                       ├─ extract_luma_from_rgb565() → 160×120 gray
                       │   → ofd_queue → ofd_task [Core 0, pri 3]
                       │        │
                       │        ▼
                       │   ofd_process_gray() [block matching]
                       │   → EMA filter → looming_detected, turn_cmd
                       │   → store_last_ofd() [portMUX protected]
                       │
                       └─ CSV row: {frame, timestamp, OFD result, IMU}
                               → s_csv_buf → flush every 4KB → V*.CSV

[SBUS RC] ──UART1──→ sbus_task [Core 1, pri 10]
                       → aileron, elevator, throttle, rudder, frequency (volatile)
                       → sw1_raw (volatile, recording switch)

[fwr_control timers]
  sin_timer (3ms): updates y = sin(ωt)
  servo_timer (4ms): left/right = 1500 + static_offsets + ofd_offsets + y×throttle
                       → LEDC GPIO4/5

[main loop reads]:
  video_rec_last_ofd() → r.looming_detected
  ail = (int)(r.turn_cmd × 600)  [computed but discarded]
  fwr_set_ofd_avoidance(ail, 0)  ← COMMENTED OUT
```

---

## AI Assistant Notes

### Safest Files to Modify
| File | Why safe |
|---|---|
| `main/ofd_config.h` | Pure constants, no logic, easy to revert |
| `video converter/*.py` | Host-side tools, no MCU impact |
| `main/main_fwr_ofd.cpp` | Well-isolated, single entry point, changes are contained |

### Dangerous Files
| File | Risk |
|---|---|
| `main/fwr_control.cpp` | ISR-context timers; wrong `y_mux` usage can corrupt servo output mid-flight |
| `main/video_rec.cpp` | Ring buffer / SPSC logic; wrong pointer arithmetic causes silent SD corruption |
| `main/ofd.cpp` | Static `u_prev_row[]` arrays are 256 elements; `maxCols > 256` path is an error return, but if ever wrong it silently corrupts stack |
| `sdkconfig` | Do not edit manually; use `idf.py menuconfig` |
| `partitions.csv` | Changing partition layout requires full flash erase |

### Likely Regression Risks
1. **Re-enabling `main.cpp`** — compile error (old OFD API), undefined `crop_to_lcd()`
2. **Removing `yolo.cpp` from SRCS** — safe, but `bas.espdl` EMBED_FILES must also be removed
3. **Enabling `fwr_set_ofd_avoidance()`** — safe API, but untested in actual flight; tune MAX_AVOID_US first
4. **Changing OFD resolution from 160×120** — static `u_prev_row[256]` buffer needs to be resized if grid produces >256 columns
5. **Adding IMU lock** — `imu_data_t` is 36 bytes; lock must use critical section, not mutex (called from main loop which may be in task context adjacent to ISR timers)

### Hidden Coupling
- `sw1_raw` is declared `volatile` in `sbus_rx.cpp` and used via `extern volatile int` in both `main_fwr_ofd.cpp` and `video_rec.cpp` — modifying SBUS channel mapping requires checking all three files
- `ofd_aileron_offset` and `ofd_rudder_offset` are volatile ints in `fwr_control.cpp` — `fwr_set_ofd_avoidance()` writes them from main loop task context; `servo_timer_cb` reads them from timer ISR context — no lock, but single-word access is atomic on ESP32
- OFD `gPrev` buffer is a module-level static in `ofd.cpp` — calling `ofd_init()` twice (e.g., at recording start) would reallocate but not free the old buffer (memory leak). Currently `video_rec.cpp:ofd_init()` is called once at boot only, so this is not triggered.

### Important Configs
- **Primary tuning file:** `main/ofd_config.h` — change thresholds here, not in `.cpp` files
- **Main selection:** `main/CMakeLists.txt` line 9 — comment/uncomment to switch entrypoints
- **Model selection:** `main/CMakeLists.txt` EMBED_FILES section — only `bas.espdl` active

### Debugging Entrypoints
- **IMU issues:** Enable `main_imu_test.cpp` — standalone I²C probe with verbose output
- **RC/SBUS issues:** Enable `main_sbus_monitor.cpp` — prints all 16 channels live
- **OFD tuning:** Use `video converter/retune_ofd.py` with recorded .MJP + .CSV — replays full pipeline offline
- **SD write speed:** `sdcard_init()` runs 4MB burst + 28MB sustained benchmark at boot — check serial log
- **Cluster size warning:** appears at boot if SD formatted with <64KB clusters

---

## Technical Debt

### Critical
1. **`s_last_imu` has no synchronization** — written from main loop (Core 0), read from writer_task (Core 1). A 9-float struct torn read will produce garbled IMU data in CSV. Fix: add `portMUX_TYPE s_imu_lock` critical section in `video_rec_set_imu()` and the writer_task read.

### Significant
2. **CSV frame coherence** — OFD results and IMU samples in a given CSV row come from different moments in time. There is no timestamp alignment between the OFD frame number and the camera frame number being encoded. Acceptable for current analysis but misleads any system that assumes tight synchronization.

3. **SPSC ring buffer lacks memory barriers** — `s_ring_wr`/`s_ring_rd` are `volatile` but not `_Atomic`. On dual-core Xtensa, `dmb` instructions are needed for true producer-consumer safety. In practice FreeRTOS task scheduling provides implicit barriers, but this is implementation-dependent.

4. **`ofd_task` and `sd_stream_task` queues are leaked on abnormal stop** — `stop_recording()` cleanly deletes queues on normal exit but has no timeout; if writer_task hangs, `stop_recording()` blocks forever.

5. **`yolo.cpp` compiled but never called** — occupies compile time, links `esp-dl` framework, and causes `bas.espdl` (100+ KB) to be embedded in flash. If YOLO is not planned for integration soon, remove from SRCS.

### Minor
6. **WDT disabled globally** (`CONFIG_ESP_TASK_WDT_EN=n`) — acceptable in lab; must be re-enabled before field tests.

7. **`ofd.cpp` `gPrev` allocation** — `malloc()` not `heap_caps_malloc(MALLOC_CAP_SPIRAM)`. At 160×120 = 19.2KB this likely comes from internal SRAM. On SPIRAM-heavy builds this competes with DMA bounce buffer. Explicit `MALLOC_CAP_SPIRAM` would be cleaner.

8. **`main.cpp` left in SRCS path** — it will not compile if re-enabled. Its presence is a trap for future developers. Should be deleted or moved to an `archive/` folder.

9. **Camera returns RGB565 but code also handles I420** — `video_rec_enqueue()` has dual-format handling (RGB565 and YU12/I420). Camera is initialized for RGB565 only, making the I420 path dead code. The I420 path also contains a buffer layout bug (slot->data is unallocated for I420 path).

10. **`esp_lcd_ek79007`** in CMakeLists REQUIRES — LCD driver linked into firmware even though LCD is not initialized. Small overhead.

---

## .claude Agent Review

Two agents are defined in `.claude/agents/`. Both use `claude-haiku-4-5` (fast, cost-effective — appropriate for search tasks).

### `esp32-reviewer.md`
- **Purpose:** RTOS/ISR/ring-buffer reviewer — run after modifying firmware
- **Issue:** Frontmatter missing `:` separators on field names. Correct format:
  ```markdown
  ---
  name: esp32-reviewer
  description: Reviews ESP32-P4 FreeRTOS firmware...
  model: claude-haiku-4-5
  tools: Read, Grep, Glob
  ---
  ```
- **Coverage:** Good for catching: priority inversions, ISR malloc, I2C errors, file handle leaks, ring buffer bounds
- **Recommendation:** Fix frontmatter format; expand to also check `volatile` races

### `codebase-explorer.md`
- **Purpose:** Search-only agent: find symbol definitions, trace usages, understand file layout
- **Issue:** Uses escaped dashes `\---` in frontmatter — will not parse correctly. Fix to plain `---`.
- **Recommendation:** Fix frontmatter format

---

## Recommended Next Steps

### 1. Immediate Fixes (< 1 hour)
- [ ] Fix `s_last_imu` race: add `portMUX_TYPE s_imu_lock` in `video_rec.cpp`, protect both read and write
- [ ] Fix `.claude/agents/esp32-reviewer.md` frontmatter (missing `:` separators)
- [ ] Fix `.claude/agents/codebase-explorer.md` frontmatter (escaped `\---`)

### 2. Stabilization (before avoidance flight)
- [ ] Uncomment `fwr_set_ofd_avoidance(ail, 0)` in `main_fwr_ofd.cpp:228`
- [ ] Add RC switch guard: only apply avoidance when a specific RC switch is ON (e.g., CH6 or CH7 in HIGH position) — prevents surprise injection during manual testing
- [ ] Add `fwr_set_ofd_avoidance(0, 0)` in the else branch (already written, just uncomment)
- [ ] Re-enable WDT for field tests (`CONFIG_ESP_TASK_WDT_EN=y`, add `esp_task_wdt_reset()` in main loop)

### 3. Validation (tethered, then free flight)
- [ ] Tethered test: verify `ofd_aileron_offset` changes produce correct servo movement direction
- [ ] Log `ail`, `r.turn_cmd`, `r.evasion_level` to serial during tethered test
- [ ] Confirm OFD correctly identifies looming from real obstacles (use retune_ofd.py on recorded data)
- [ ] Tune `MAX_AVOID_US` (start at 200, not 600 — prevents aggressive roll)
- [ ] Tune `OFD_LR_GAIN` (currently 3.0 — may oversteer)
- [ ] Tune `OFD_FLOW_THRESH_BRAKE` (currently 3.5 px/frame — validate against real flight data)

### 4. Autonomous Flight Enablement
- [ ] Use `evasion_level` for graduated response instead of binary `looming_detected`
- [ ] Add hysteresis: do not clear avoidance command until `ema_flow_mag < OFD_FLOW_THRESH_ALERT`
- [ ] Consider enabling `OFD_USE_AZ_QUIET_GATE` for real flapping flight (removes wing-vibration frames)
- [ ] Add rudder mixing: `fwr_set_ofd_avoidance(ail_us, rud_us)` for coordinated turns

### 5. Future Architecture Improvements
- [ ] Remove `yolo.cpp` from active SRCS or integrate it into the detection pipeline
- [ ] Delete or archive `main.cpp` (broken, misleading)
- [ ] Fix SPSC ring buffer: use `_Atomic` or add explicit memory barriers
- [ ] Add per-frame OFD frame number to CSV to enable proper frame-aligned analysis
- [ ] Consider EKF for pose fusion (currently no implementation exists — would be new work)

---

## Quick Onboarding

### For AI Agents

**The one active entrypoint is `main/main_fwr_ofd.cpp`.** Everything starts there.

**The one tuning file is `main/ofd_config.h`.** Change thresholds there, not in `.cpp` files.

**The avoidance is one uncomment away.** Line 228 of `main_fwr_ofd.cpp`. The servo infrastructure (`fwr_set_ofd_avoidance`) is fully implemented.

**Never re-enable `main.cpp`** — it will not compile (old 1-arg OFD API, missing `crop_to_lcd`).

**EKF does not exist** — any reference to `ofd_ekf.cpp` is incorrect.

**Before editing `fwr_control.cpp`**: remember `sin_timer_cb` and `servo_timer_cb` run in timer ISR context. Any shared variable they touch needs `portMUX_TYPE` critical section protection.

### For Developers

```bash
# Flash and monitor
idf.py -p COM{port} -b 921600 flash monitor

# Convert recorded video (on host PC, in 'video converter/' folder)
python convert_mjp.py V0001.MJP

# Re-tune OFD parameters offline
python retune_ofd.py V0001.MJP --alpha 0.4 --flow-brake 3.0

# Enable avoidance (one change needed):
# Edit main/main_fwr_ofd.cpp line 228:
#   uncomment: fwr_set_ofd_avoidance(ail, 0);
#   uncomment: fwr_set_ofd_avoidance(0, 0);
```

### Key File Locations

| What you need | Where to look |
|---|---|
| OFD thresholds | `main/ofd_config.h` |
| Active main | `main/main_fwr_ofd.cpp` |
| Avoidance injection | `main_fwr_ofd.cpp:223-231` |
| Servo mixing | `fwr_control/fwr_control.cpp:115-138` |
| RC channel mapping | `fwr_control/sbus_rx.cpp:112-155` |
| OFD trigger logic | `video_rec.cpp:1017-1036` |
| IMU wiring | `main_fwr_ofd.cpp:27-30` |
| SD card wiring | `video_rec.cpp:468-473` |
| Task priorities | `video_rec.cpp:671-682` + `sbus_rx.cpp:197-205` |

---

## Current Readiness Assessment

| Category | Rating | Evidence |
|---|---|---|
| **Vision** (camera + downsampling) | **functional** | OV5647 RGB565 at 1280×960 confirmed working; grayscale downsampling to 160×120 implemented and tested |
| **Detection** (OFD pipeline) | **functional** | Tuned on 1536 real frames (V0000.CSV); bias (-0.018), EMA (α=0.30), flow/div dual-gate all calibrated |
| **Control** (servo + RC) | **stable** | SBUS decode, sin-wave stroke, servo mixing all confirmed working in flight |
| **Autonomy** (avoidance loop) | **partial** | Detection→decision chain complete; servo injection commented out; mixing infrastructure ready |
| **Synchronization** | **partial** | `s_last_ofd` protected; `s_last_imu` has confirmed race; CSV frame coherence structurally loose |
| **Logging** | **functional** | MJPEG + CSV sidecar working; post-processing tools functional; coherence imperfect |
| **Maintainability** | **prototype** | Dead code (`main.cpp`, unused models); broken agent frontmatter; no tests; WDT disabled |
| **Flight Readiness** | **partial** | Manual flight: ready. Autonomous avoidance: needs 1 code change + tethered validation |
