# Depth-Guided Computer Vision for Precision Agricultural Fertilizer Application

**Project page → https://ivantorriani.github.io/tsmg-2026/**

A real-time monocular depth and canopy detection pipeline that drives zone-resolved fertilizer
nozzle actuation on NVIDIA Jetson edge hardware.

A single RGB frame is passed through MiDaS DPT to obtain a relative inverse-depth map, which is
optionally used to suppress background clutter before a YOLOv8 canopy detector runs. Detections
are collapsed into a binary union mask, integrated over three horizontal zones aligned with the
physical nozzle bar, and thresholded at 25% coverage to produce three independent GPIO signals.

**Ivan Torriani** — Lead Researcher &nbsp;·&nbsp; **Dr. Fahim Khan** — Principal Investigator

## Repository layout

| Path | Contents |
|---|---|
| `src/midas_scripts/` | Perception pipeline. `run_combined.py` is the deployed live-camera entry point. |
| `src/led_scripts/` | Actuation and hardware validation. `ledtest*.py` are the GPIO bring-up scripts. |
| `docs/methods_midas.md` | Methods: depth estimation, calibration, zone analysis. |
| `docs/methods_led.md` | Methods: GPIO platform, validation progression, actuation mapping. |
| `index.html`, `assets/` | Project page served by GitHub Pages from the repository root. |

## Running the live pipeline

```bash
pip install -r requirements.txt

python src/midas_scripts/run_combined.py \
    --yolo_weights canopies_latest.pt \
    --model_type dpt_swin2_tiny_256 \
    --optimize \
    --fuse_yolo \
    --show_depth_side
```

Add `--no_gpio` to run off-Jetson hardware. `Jetson.GPIO` is only installable on the Jetson
itself; the pipeline detects its absence and disables GPIO output rather than failing.

Key flags: `--coverage_threshold` (default `0.25`) sets the zone activation threshold,
`--use_depth_mask` enables depth-based background suppression, and `--detect_on_depth_mask`
feeds the masked frame to the detector instead of the raw one.

## Status

The methods and system design are complete and documented. Quantitative evaluation — throughput
on target hardware, per-zone activation accuracy, and calibration error across a distance sweep —
is still being collected; those tables are scaffolded on the project page and marked `TODO`.
