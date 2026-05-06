# Methods: GPIO-Based Nozzle Activation System (LED Scripts)

## Overview

This section describes the hardware output subsystem responsible for translating spatial canopy detection results into physical actuation signals. The system drives two GPIO output pins on an NVIDIA Jetson platform, which serve as control signals for nozzle activation in the field application. Development proceeded through iterative hardware validation stages before integration with the full detection pipeline.

---

## 3.1 Hardware Platform and GPIO Interface

All GPIO control was implemented on an NVIDIA Jetson single-board computer. Two GPIO libraries were evaluated during development:

- **`gpiod` (libgpiod)** — a character-device-based GPIO interface used in early validation scripts. Lines were requested by chip name (`/dev/gpiochip0`) and offset number, with direction and initial output value specified at request time.
- **`Jetson.GPIO`** — the Jetson-specific GPIO library, used in all subsequent and production scripts. Pin numbering followed the BOARD convention, which maps to physical connector pin positions rather than SoC GPIO numbers.

Two output pins were used throughout the system:

| Signal | BOARD Pin | Purpose |
|--------|-----------|---------|
| `led_pin1` | Pin 7 | Nozzle control output 1 |
| `led_pin2` | Pin 33 | Nozzle control output 2 |

Pins were initialized to LOW at startup. GPIO cleanup (setting all pins LOW and releasing resources) was performed on exit to prevent undefined hardware states.

---

## 3.2 GPIO Validation and Iterative Development

Prior to integration with the detection pipeline, a series of standalone GPIO validation scripts were developed to verify hardware connectivity and signal behavior:

**Table 2. GPIO validation script progression.**

| Stage | Script | Library | Pins Used | Behavior | Purpose |
|---|---|---|---|---|---|
| 1 | `led_test2.py` | `gpiod` (v2 API) | 1 (offset 144) | Single HIGH/LOW toggle, 2 s intervals | Confirm basic electrical continuity |
| 2 | `ledtest3.py` | `gpiod` (legacy API) | 1 (line 105) | Continuous 1 s toggle loop | Verify sustained operation |
| 3 | `ledtest4.py` | `Jetson.GPIO` | 1 (BOARD pin 7) | Continuous 1 s toggle loop | Confirm Jetson.GPIO compatibility and BOARD pin convention |
| 4 | `ledtest5.py` | `Jetson.GPIO` | 2 (pins 7, 33) | Keyboard-driven activation via `curses` | Validate independent two-channel control |

---

## 3.3 Zone-to-Pin Activation Mapping

The nozzle activation logic maps the three spatial detection zones (top, middle, bottom) to two GPIO output pins using a combinatorial encoding scheme. This encoding was designed to represent three independent spatial states using two binary output lines:

**Table 3. Zone-to-pin activation truth table (single-zone active).**

| Zone Active | Pin 1 (BOARD 7) | Pin 2 (BOARD 33) |
|---|---|---|
| Top | HIGH | HIGH |
| Middle | HIGH | LOW |
| Bottom | LOW | HIGH |
| None | LOW | LOW |

When multiple zones are simultaneously active, pin states are resolved as logical ORs across all contributing zones:

**Table 4. Multi-zone activation states.**

| Top | Middle | Bottom | Pin 1 | Pin 2 |
|---|---|---|---|---|
| ✗ | ✗ | ✗ | LOW | LOW |
| ✓ | ✗ | ✗ | HIGH | HIGH |
| ✗ | ✓ | ✗ | HIGH | LOW |
| ✗ | ✗ | ✓ | LOW | HIGH |
| ✓ | ✓ | ✗ | HIGH | HIGH |
| ✓ | ✗ | ✓ | HIGH | HIGH |
| ✗ | ✓ | ✓ | HIGH | HIGH |
| ✓ | ✓ | ✓ | HIGH | HIGH |

The Boolean expressions governing pin state are:

```
pin1_high = top_active OR middle_active
pin2_high = top_active OR bottom_active
```

The final pin states are written in a single pass to avoid transient conflicts that could arise from sequential writes when multiple zones are active.

---

## 3.4 Integration with Detection Pipeline

In the integrated system (`nozzle_sim_led.py`, `run_combined.py`), GPIO output was driven once per frame immediately after zone activity was determined. The activation logic operated as follows:

1. The detection union mask was computed from all YOLOv8 bounding boxes for the target class within the current frame.
2. The frame was partitioned into three equal horizontal zones, and the coverage ratio was computed for each zone.
3. Zones with coverage exceeding 25% were marked Active.
4. GPIO pin states were updated according to the mapping table above.

In the offline video processing mode (`nozzle_sim_led.py`), frames were read sequentially from a pre-recorded input video, allowing the activation logic to be validated against known footage before field deployment. The output video, with annotated zone boundaries and activation states, was written to disk for post-hoc review.

In the live camera mode (`run_combined.py`), the same logic operated on frames captured in real time from the camera, with GPIO outputs driving physical hardware on each frame.

GPIO initialization and cleanup were encapsulated in dedicated functions (`setup_gpio`, `cleanup_gpio`) to ensure safe hardware state management across normal exits and exceptions.

---

## 3.5 Annotated Output

For each processed frame, the activation state of each zone was rendered visually:

- Active zones were highlighted with a semi-transparent green overlay (α = 0.3) and a green bounding rectangle.
- Inactive zones were outlined in red.
- Zone labels ("Nozzle 1", "Nozzle 2", "Nozzle 3"), activation status, and per-zone coverage percentages were rendered as text overlays.

**Table 5. Annotated output video parameters.**

| Parameter | Value | Notes |
|---|---|---|
| Video codec | XVID / mp4v | XVID used in offline mode; mp4v in live mode |
| Container format | MP4 | `.mp4` extension |
| Output frame rate | Source FPS (offline) / 20 FPS (live) | Matches input video or fixed for live capture |
| Active zone overlay color | Green (BGR: 0, 255, 0) | Semi-transparent fill, α = 0.3 |
| Inactive zone border color | Red (BGR: 0, 0, 255) | Outline only |
| Detection box color | Blue (BGR: 255, 0, 0) | Drawn around each YOLO bounding box |
| Display preview resolution | 1280 px wide (live) / 1280 × 1280 (offline) | Resized for monitor display only; recorded at native resolution |

This annotated output was written to disk for each experimental run, providing a complete record of detection and activation events for post-hoc review.
