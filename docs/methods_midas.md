# Methods: Monocular Depth Estimation Pipeline (MiDaS Scripts)

## Overview

This section describes the monocular depth estimation pipeline developed for real-time canopy proximity detection. The system integrates a pre-trained depth estimation network with a custom-trained object detection model to produce per-frame depth maps and spatial activation signals. All processing runs on an NVIDIA Jetson platform with GPU acceleration.

---

## 2.1 Depth Estimation Model

Depth estimation was performed using MiDaS (Mixed Dataset Sampling), a monocular depth estimation framework that produces relative inverse depth maps from single RGB images [Ranftl et al., 2020; Ranftl et al., 2022]. The Dense Prediction Transformer (DPT) architecture was used, which couples a transformer-based encoder with a convolutional decoder to produce dense, high-resolution depth predictions.

Two model variants were evaluated during development:

| Model Identifier | Backbone | Input Resolution | Aspect Ratio Preserved | Role |
|---|---|---|---|---|
| `dpt_beit_large_512` | BEiT ViT-L/16 | 512 × 512 | Yes | High-accuracy reference |
| `dpt_swin2_tiny_256` | Swin Transformer V2 Tiny | 256 × 256 | No (square crop) | Jetson deployment |

The DPT architecture extracts multi-scale feature representations from four intermediate transformer layers and fuses them through a series of Reassemble and Fusion (RefineNet) blocks. The decoder progressively upsamples these fused features and applies a convolutional head to produce a single-channel inverse depth map. The output is constrained to be non-negative via a ReLU activation.

Input images were preprocessed using a composed transform pipeline and converted to the CHW tensor format expected by the network. Inference was performed under `torch.no_grad()` to suppress gradient computation. On CUDA hardware, optional FP16 (half-float) optimization was applied via `model.half()` and `torch.channels_last` memory layout to reduce memory bandwidth and improve throughput.

**Table 1. Input preprocessing parameters for DPT model variants.**

| Step | Parameter | Value |
|---|---|---|
| Resize method | Mode | Minimal (pad to multiple of 32) |
| Normalization | Per-channel mean | [0.5, 0.5, 0.5] |
| Normalization | Per-channel std | [0.5, 0.5, 0.5] |
| Output upsample | Interpolation mode | Bicubic |
| FP16 optimization | Memory layout | `torch.channels_last` |
| Gradient computation | Context | `torch.no_grad()` |

The raw network output was upsampled back to the original frame resolution using bicubic interpolation (`torch.nn.functional.interpolate` with `mode="bicubic"`) to produce a full-resolution inverse depth map.

---

## 2.2 Depth-to-Distance Calibration

Because MiDaS produces relative (affine-invariant) inverse depth values rather than metric distances, a calibration procedure was performed to establish a correspondence between model output and physical distance.

A reference object was placed at a known distance of 2 feet from the camera. The maximum depth value across the frame (`dmax`) was recorded at this distance, yielding a calibration constant:

```
SCENE_1_SHIFT = 0.00176834659   (= 2 ft / dmax_at_2ft)
```

A first-order metric estimate was computed as:

```
metric_prediction = SCENE_1_SHIFT × dmax
```

To account for the non-linear relationship between inverse depth and physical distance, a refined estimate was derived using an inverse power-law formulation inspired by the depth-distance relationship D = 1/(ad + b):

```
refined_metric_prediction = 2.0 × (dmax_scene_1 / dmax)^sensitivity
```

where `dmax_scene_1 = 1130.5822` is the calibrated reference depth value and `sensitivity = 3.2` is an empirically tuned exponent controlling the rate of distance change with depth. This formulation provides improved resolution at close range, where the system is most sensitive to canopy proximity.

**Table 2. Depth-to-distance calibration constants.**

| Parameter | Symbol | Value | Description |
|---|---|---|---|
| Scene shift constant | `SCENE_1_SHIFT` | 0.00176834659 | Linear scale factor (ft / depth unit), derived at 2 ft reference |
| Reference depth value | `dmax_scene_1` | 1130.5822 | Maximum depth output recorded at 2 ft calibration distance |
| Reference distance | — | 2.0 ft | Physical distance used for calibration |
| Sensitivity exponent | `sensitivity` | 3.2 | Power-law exponent for refined distance estimate |

A binary proximity conclusion was derived from the first-order estimate: when the reflected metric value fell below zero (indicating the object was closer than the calibration reference), the frame was labeled as "Object Detected Close to Camera."

---

## 2.3 Background Segmentation via Depth Masking

To reduce spurious object detections from background elements, a depth-based foreground segmentation step was applied prior to object detection. Pixels with inverse depth values below a fixed threshold (`depth_threshold = 1130.5822`) were classified as background. The resulting binary mask was refined using morphological operations to remove noise and fill small holes. Background pixels were replaced with a solid black fill (RGB = [0, 0, 0]), isolating the foreground canopy region for downstream processing.

**Table 3. Depth masking parameters.**

| Parameter | Value | Purpose |
|---|---|---|
| Depth threshold | 1130.5822 | Boundary between foreground and background in inverse depth units |
| Morphological kernel | 5 × 5 (ones) | Structuring element for opening and closing |
| Morphological opening | Applied first | Removes small foreground noise |
| Morphological closing | Applied second | Fills small holes in foreground mask |
| Background fill color | RGB = [0, 0, 0] | Replaces background pixels with black |

This masking step was configurable: it could be applied to the display output only, or used as the input to the object detector, depending on experimental conditions.

---

## 2.4 Object Detection

Canopy detection was performed using a YOLOv8 model (Ultralytics) fine-tuned on a custom dataset of canopy images (`canopies_latest.pt`). The model was loaded onto the GPU and optimized using layer fusion (`model.fuse()`) and FP16 inference (`model.half()`) to maximize throughput.

**Table 4. YOLOv8 object detection configuration.**

| Parameter | Value | Description |
|---|---|---|
| Weights file | `canopies_latest.pt` | Custom-trained canopy detection model |
| Confidence threshold | 0.5 | Minimum score to accept a detection |
| Target class | 0 (canopy) | Only detections of this class are used |
| Inference precision | FP16 | Half-float on CUDA for throughput |
| Layer fusion | Enabled | `model.fuse()` merges Conv+BN layers |
| Device | CUDA (GPU 0) | Falls back to CPU if unavailable |

For each detected bounding box, the pixel coordinates were extracted and used to construct a binary union mask over the frame, marking all pixels within any detected bounding box as foreground detections.

---

## 2.5 Spatial Zone Analysis

The frame was divided into three equal horizontal bands, referred to as nozzle zones:

**Table 5. Nozzle zone definitions.**

| Zone | Label | Row Range | Fraction of Frame Height |
|---|---|---|---|
| 1 | Top | 0 to H/3 | Upper third |
| 2 | Middle | H/3 to 2H/3 | Middle third |
| 3 | Bottom | 2H/3 to H | Lower third |

For each zone, the coverage ratio was computed as the fraction of pixels within the zone that fell inside at least one detection bounding box:

```
coverage_ratio_i = (detected pixels in zone i) / (total pixels in zone i)
```

A zone was marked **Active** if its coverage ratio exceeded a threshold of 0.25 (25%). This threshold was chosen empirically to balance sensitivity against false activations from partial or distant canopy detections.

---

## 2.6 Real-Time Inference Loop

The full pipeline was implemented as a continuous frame-processing loop using `imutils.VideoStream` for low-latency camera capture. Each iteration performed the following steps in sequence:

**Table 6. Per-frame processing pipeline.**

| Step | Operation | Notes |
|---|---|---|
| 1 | Camera frame capture | BGR format via `VideoStream` |
| 2 | Color conversion and normalization | BGR → RGB, scaled to [0, 1] |
| 3 | MiDaS depth inference | Output upsampled to original resolution via bicubic interpolation |
| 4 | Depth-based background masking | Optional; controlled by `--use_depth_mask` flag |
| 5 | YOLOv8 canopy detection | Run on raw or depth-masked frame |
| 6 | Detection union mask construction | Binary mask of all bounding box pixels |
| 7 | Per-zone coverage computation | Coverage ratio for each of three horizontal zones |
| 8 | Zone activation and GPIO output | Pins driven based on zone states |
| 9 | Annotated frame rendering | Zone overlays, labels, coverage percentages |
| 10 | Optional video write | Frame appended to output MP4 |

Frame rate was monitored using an exponential moving average (EMA) with α = 0.1. CuDNN benchmark mode (`torch.backends.cudnn.benchmark = True`) was enabled to allow the runtime to select optimized convolution algorithms for the fixed input resolution.

---

## 2.7 Visualization

For diagnostic purposes, a side-by-side visualization was generated by placing the annotated RGB frame alongside a normalized depth heatmap. The depth map was linearly normalized to [0, 255] and rendered using the INFERNO colormap (OpenCV `COLORMAP_INFERNO`), where brighter values correspond to closer objects. Zone boundaries, activation states, and per-zone coverage percentages were overlaid on the RGB frame using colored bounding rectangles (green = Active, red = Inactive) and semi-transparent shading (α = 0.3) for active zones.

---

## References

- Ranftl, R., Lasinger, K., Hafner, D., Schindler, K., & Koltun, V. (2020). Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.
- Ranftl, R., Bochkovskiy, A., & Koltun, V. (2021). Vision Transformers for Dense Prediction. *ICCV 2021*.
