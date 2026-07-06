# -*- coding: utf-8 -*-
"""
Live camera nozzle activation with YOLO object detection, MiDaS depth estimation, and Jetson GPIO output.

What this script does:
1. Reads frames from a live camera.
2. Runs MiDaS monocular depth estimation.
3. Optionally masks/blackens background using the depth map.
4. Runs YOLO object detection on the current frame.
5. Splits the frame into three horizontal nozzle zones:
      Nozzle 1 = top third
      Nozzle 2 = middle third
      Nozzle 3 = bottom third
6. Marks each zone Active if detected canopy/object boxes cover more than a threshold percentage of that zone.
7. Drives GPIO outputs according to active zones.
8. Shows and optionally records an annotated preview video.

Example:
    python depth_nozzle_gpio_live.py \
        --yolo_weights model0.pt \
        --model_type dpt_swin2_tiny_256 \
        --output_path output_vids \
        --show_depth_side

For Jetson GPIO, run on the Jetson with Jetson.GPIO installed.
"""

import argparse
import os
import time
from typing import List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from imutils.video import VideoStream
from midas.model_loader import default_models, load_model
import utils

try:
    import Jetson.GPIO as GPIO
    GPIO_AVAILABLE = True
except Exception:
    GPIO_AVAILABLE = False
    GPIO = None


# ----------------------------
# Calibration constants
# ----------------------------
SCENE_1_SHIFT = 0.00176834659
DEFAULT_DEPTH_THRESHOLD = 1130.5822


# ----------------------------
# Depth utilities
# ----------------------------
def solid_background(
    img_rgb: np.ndarray,
    depth_map: np.ndarray,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    depth_threshold: float = DEFAULT_DEPTH_THRESHOLD,
) -> np.ndarray:
    """
    Replace background behind a depth threshold with a solid RGB color.

    MiDaS depth is relative, not metric. In your previous script, lower values were treated
    as farther/background and higher values as closer/foreground.
    """
    if depth_threshold is None:
        d_min, d_max = float(depth_map.min()), float(depth_map.max())
        depth_threshold = d_min + 0.6 * (d_max - d_min)

    background_mask = (depth_map < depth_threshold).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_OPEN, kernel)
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_CLOSE, kernel)

    foreground_mask = 1 - background_mask
    bg = np.full_like(img_rgb, bg_color, dtype=img_rgb.dtype)
    result_rgb = np.where(foreground_mask[..., None] == 1, img_rgb, bg)
    return result_rgb


def create_side_by_side(image_bgr: np.ndarray, depth: np.ndarray, grayscale: bool) -> np.ndarray:
    """Create BGR image beside normalized depth visualization."""
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    denom = max(depth_max - depth_min, 1e-6)

    normalized_depth = 255.0 * (depth - depth_min) / denom
    right_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2)
    right_side = np.uint8(np.clip(right_side, 0, 255))

    if not grayscale:
        right_side = cv2.applyColorMap(right_side, cv2.COLORMAP_INFERNO)

    if image_bgr is None:
        return right_side
    return np.concatenate((image_bgr, right_side), axis=1)


def run_midas_process(
    device: torch.device,
    model,
    model_type: str,
    image: np.ndarray,
    input_size: Tuple[int, int],
    target_size: Tuple[int, int],
    optimize: bool,
) -> np.ndarray:
    """Run one MiDaS inference pass and resize prediction to target_size."""
    if "openvino" in model_type:
        sample = [np.reshape(image, (1, 3, *input_size))]
        prediction = model(sample)[model.output(0)][0]
        prediction = cv2.resize(prediction, dsize=target_size, interpolation=cv2.INTER_CUBIC)
        return prediction

    sample = torch.from_numpy(image).to(device).unsqueeze(0)

    if optimize and device.type == "cuda":
        sample = sample.to(memory_format=torch.channels_last).half()

    prediction = model.forward(sample)
    prediction = (
        torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=target_size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )
    return prediction


# ----------------------------
# GPIO utilities
# ----------------------------
def setup_gpio(enabled: bool, led_pin1: int, led_pin2: int, led_pin3: int) -> bool:
    """Initialize GPIO if requested and available."""
    if not enabled:
        print("GPIO disabled by argument.")
        return False

    if not GPIO_AVAILABLE:
        print("Warning: Jetson.GPIO is not available. GPIO output will be disabled.")
        return False

    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(led_pin1, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(led_pin2, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(led_pin3, GPIO.OUT, initial=GPIO.LOW)
    print(f"GPIO enabled. led_pin1={led_pin1}, led_pin2={led_pin2}, led_pin3={led_pin3}")
    return True


def write_gpio_states(gpio_enabled: bool, led_pin1: int, led_pin2: int, led_pin3: int, region_active: List[bool]) -> None:
    """
    Drive three GPIO pins, one per zone.

    Mapping:
      top active    -> led_pin1 HIGH
      middle active -> led_pin2 HIGH
      bottom active -> led_pin3 HIGH
    """
    if not gpio_enabled:
        return

    top_active, middle_active, bottom_active = region_active

    GPIO.output(led_pin1, GPIO.HIGH if top_active else GPIO.LOW)
    GPIO.output(led_pin2, GPIO.HIGH if middle_active else GPIO.LOW)
    GPIO.output(led_pin3, GPIO.HIGH if bottom_active else GPIO.LOW)


def cleanup_gpio(gpio_enabled: bool, led_pin1: int, led_pin2: int, led_pin3: int) -> None:
    if not gpio_enabled:
        return
    try:
        GPIO.output(led_pin1, GPIO.LOW)
        GPIO.output(led_pin2, GPIO.LOW)
        GPIO.output(led_pin3, GPIO.LOW)
        GPIO.cleanup()
    except Exception as exc:
        print(f"GPIO cleanup warning: {exc}")


# ----------------------------
# Object detection and nozzle-zone utilities
# ----------------------------
def build_detection_mask_and_draw(
    frame_bgr: np.ndarray,
    yolo_results,
    class_id: int,
    box_color: Tuple[int, int, int] = (255, 0, 0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a union mask from YOLO boxes of a target class and draw those boxes.

    Returns:
      det_mask: uint8 HxW mask with 1 inside boxes
      annotated: BGR frame with detection boxes drawn
    """
    h, w = frame_bgr.shape[:2]
    det_mask = np.zeros((h, w), dtype=np.uint8)
    annotated = frame_bgr.copy()

    for result in yolo_results:
        if result.boxes is None:
            continue

        for conf, cls, bbox in zip(result.boxes.conf, result.boxes.cls, result.boxes.xyxy):
            cls_int = int(cls.item())
            if cls_int != class_id:
                continue

            x1, y1, x2, y2 = bbox.detach().cpu().numpy().astype(int).tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            cv2.rectangle(det_mask, (x1, y1), (x2 - 1, y2 - 1), 1, thickness=-1)

            label = f"Object: {float(conf.item()) * 100:.1f}%"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3)
            cv2.putText(
                annotated,
                label,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                box_color,
                2,
            )

    return det_mask, annotated


def compute_region_activity(
    det_mask: np.ndarray,
    coverage_threshold: float,
) -> Tuple[List[bool], List[float], List[int]]:
    """
    Split the detection mask into three horizontal bands and compute activity.

    Returns:
      region_active: three booleans
      coverage_ratios: percentage coverage for each region, in [0, 1]
      y_edges: four y coordinates defining the three bands
    """
    h, w = det_mask.shape[:2]
    y_edges = [0, int(round(h / 3)), int(round(2 * h / 3)), h]

    region_active = [False, False, False]
    coverage_ratios = [0.0, 0.0, 0.0]

    for i in range(3):
        y1_area = y_edges[i]
        y2_area = y_edges[i + 1]
        region_h = max(0, y2_area - y1_area)
        if region_h == 0:
            continue

        region_area = region_h * w
        covered_pixels = int(det_mask[y1_area:y2_area, :].sum())
        coverage_ratio = covered_pixels / max(region_area, 1)

        coverage_ratios[i] = coverage_ratio
        region_active[i] = coverage_ratio > coverage_threshold

    return region_active, coverage_ratios, y_edges


def draw_nozzle_regions(
    frame_bgr: np.ndarray,
    region_active: List[bool],
    coverage_ratios: List[float],
    y_edges: List[int],
    alpha: float = 0.3,
) -> np.ndarray:
    """Draw nozzle zones, active shading, labels, and coverage ratios."""
    annotated = frame_bgr.copy()
    h, w = annotated.shape[:2]
    labels = ["Nozzle 1", "Nozzle 2", "Nozzle 3"]

    for i in range(3):
        y1 = y_edges[i]
        y2 = y_edges[i + 1] - 1
        active = region_active[i]

        if active:
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, y1), (w - 1, y2), (0, 255, 0), -1)
            cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0, annotated)

        color = (0, 255, 0) if active else (0, 0, 255)
        status = "Active" if active else "Inactive"
        coverage_pct = coverage_ratios[i] * 100.0

        cv2.rectangle(annotated, (0, y1), (w - 1, y2), color, 3)
        cv2.putText(
            annotated,
            f"{labels[i]} {status} - {coverage_pct:.1f}%",
            (10, max(35, y1 + 40)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )

    return annotated


def draw_depth_text(
    frame_bgr: np.ndarray,
    prediction: np.ndarray,
    dmax_scene_1: float,
    sensitivity: float,
) -> np.ndarray:
    """Draw the relative depth diagnostics from your second script."""
    annotated = frame_bgr.copy()
    dmax = float(np.max(prediction))

    metric_prediction = SCENE_1_SHIFT * dmax
    refined_metric_prediction = 2.0 * (dmax_scene_1 / max(dmax, 1e-6)) ** sensitivity

    if metric_prediction > 2:
        metric_prediction = 2 - abs(2 - metric_prediction)
    elif metric_prediction < 2:
        metric_prediction = 2 + abs(2 - metric_prediction)

    if isinstance(metric_prediction, float) and metric_prediction < 0:
        conclusion = "Object Detected Close to Camera"
    else:
        conclusion = "No Objects Detected"

    cv2.putText(
        annotated,
        f"Depth dmax: {dmax:.2f}",
        (35, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"Refined distance approx: {refined_metric_prediction:.2f} ft | {conclusion}",
        (35, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated


# ----------------------------
# Main live loop
# ----------------------------
def run_live(args: argparse.Namespace) -> None:
    print("Initialize")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print("GPU connected successfully")

    gpio_enabled = setup_gpio(
        enabled=not args.no_gpio,
        led_pin1=args.led_pin1,
        led_pin2=args.led_pin2,
        led_pin3=args.led_pin3,
    )

    # Load YOLO model.
    print(f"Loading YOLO weights: {args.yolo_weights}")
    od_model = YOLO(args.yolo_weights)

    if device.type == "cuda":
        od_model.to("cuda")
        if args.fuse_yolo:
            od_model.fuse()

    # Load MiDaS model.
    print(f"Loading MiDaS model: {args.model_type}")
    model, transform, net_w, net_h = load_model(
        device,
        args.model_weights,
        args.model_type,
        args.optimize,
        args.height,
        args.square,
    )

    if args.optimize and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last).half()

    if args.output_path is not None:
        os.makedirs(args.output_path, exist_ok=True)

    video = VideoStream(src=args.camera_index).start()
    time.sleep(args.camera_warmup)

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)

    video_writer = None
    frame_index = 0
    fps_ema = 30.0
    last_time = time.time()

    print("Start processing. Press ESC to exit.")

    try:
        with torch.no_grad():
            while True:
                frame_bgr = video.read()
                if frame_bgr is None:
                    print("Warning: empty frame from camera.")
                    continue

                original_image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                midas_input = transform({"image": original_image_rgb / 255.0})["image"]

                # Depth prediction at original frame size.
                prediction = run_midas_process(
                    device=device,
                    model=model,
                    model_type=args.model_type,
                    image=midas_input,
                    input_size=(net_w, net_h),
                    target_size=original_image_rgb.shape[1::-1],
                    optimize=args.optimize,
                )

                # Use depth to generate the object-detection input/preview background.
                if args.use_depth_mask:
                    depth_masked_rgb = solid_background(
                        original_image_rgb,
                        prediction,
                        bg_color=(0, 0, 0),
                        depth_threshold=args.depth_threshold,
                    )
                    depth_masked_bgr = cv2.cvtColor(depth_masked_rgb, cv2.COLOR_RGB2BGR)
                else:
                    depth_masked_bgr = frame_bgr.copy()

                # YOLO detection.
                # Use depth-masked image if desired; this can reduce background detections.
                yolo_input = depth_masked_bgr if args.detect_on_depth_mask else frame_bgr
                yolo_results = od_model.predict(
                    source=yolo_input,
                    conf=args.conf,
                    classes=[args.class_id],
                    device=0 if device.type == "cuda" else "cpu",
                    half=(device.type == "cuda"),
                    verbose=False,
                )

                det_mask, detection_annotated = build_detection_mask_and_draw(
                    depth_masked_bgr,
                    yolo_results,
                    class_id=args.class_id,
                )

                region_active, coverage_ratios, y_edges = compute_region_activity(
                    det_mask,
                    coverage_threshold=args.coverage_threshold,
                )

                write_gpio_states(
                    gpio_enabled,
                    args.led_pin1,
                    args.led_pin2,
                    args.led_pin3,
                    region_active,
                )

                content = draw_nozzle_regions(
                    detection_annotated,
                    region_active,
                    coverage_ratios,
                    y_edges,
                )

                if args.show_depth_text:
                    content = draw_depth_text(
                        content,
                        prediction,
                        dmax_scene_1=args.depth_threshold,
                        sensitivity=args.sensitivity,
                    )

                if args.show_depth_side:
                    content = create_side_by_side(content, prediction, grayscale=args.grayscale)

                # Initialize writer once using actual output dimensions.
                if args.record and video_writer is None:
                    h, w = content.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
                    timestamp = int(time.time())
                    save_path = os.path.join(args.output_path or ".", f"depth_nozzle_output_{timestamp}.mp4")
                    video_writer = cv2.VideoWriter(save_path, fourcc, args.output_fps, (w, h))
                    print(f"Recording to: {save_path}")

                if video_writer is not None:
                    video_writer.write(content)

                # Optional resize for display only.
                display = content
                if args.display_width is not None:
                    h, w = display.shape[:2]
                    scale = args.display_width / float(w)
                    display = cv2.resize(display, (args.display_width, int(h * scale)))

                cv2.imshow(args.window_name, display)

                now = time.time()
                dt = max(now - last_time, 1e-6)
                fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt)
                last_time = now

                if frame_index % args.print_every == 0:
                    print(
                        f"Frame {frame_index} | FPS {fps_ema:.2f} | "
                        f"active={region_active} | "
                        f"coverage={[round(x * 100, 1) for x in coverage_ratios]}%",
                        flush=True,
                    )

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break

                frame_index += 1

    finally:
        print("Cleaning up")
        video.stop()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        cleanup_gpio(gpio_enabled, args.led_pin1, args.led_pin2, args.led_pin3)
        print("Finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live MiDaS depth + YOLO three-zone nozzle detection + Jetson GPIO output"
    )

    # YOLO / object detection
    parser.add_argument("--yolo_weights", default="model0.pt", help="YOLO weights path, e.g. model0.pt")
    parser.add_argument("--class_id", type=int, default=0, help="YOLO class ID to use for nozzle activation")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold")
    parser.add_argument("--coverage_threshold", type=float, default=0.25, help="Region active threshold; 0.25 means 25 percent coverage")
    parser.add_argument("--fuse_yolo", action="store_true", help="Fuse YOLO model layers on CUDA")

    # MiDaS / depth
    parser.add_argument("-m", "--model_weights", default=None, help="MiDaS model weights path")
    parser.add_argument(
        "-t",
        "--model_type",
        default="dpt_swin2_tiny_256",
        help="MiDaS model type. Use dpt_swin2_tiny_256 for better Jetson speed, or dpt_beit_large_512 for quality.",
    )
    parser.add_argument("--optimize", action="store_true", help="Use half-float optimization on CUDA")
    parser.add_argument("--height", type=int, default=None, help="Preferred MiDaS encoder height")
    parser.add_argument("--square", action="store_true", help="Resize MiDaS input to square resolution")
    parser.add_argument("--grayscale", action="store_true", help="Use grayscale depth visualization")
    parser.add_argument("--depth_threshold", type=float, default=DEFAULT_DEPTH_THRESHOLD, help="Depth threshold for foreground/background masking")
    parser.add_argument("--sensitivity", type=float, default=3.2, help="Depth-to-distance sensitivity exponent")
    parser.add_argument("--use_depth_mask", action="store_true", help="Blacken background using MiDaS before display/annotation")
    parser.add_argument("--detect_on_depth_mask", action="store_true", help="Run YOLO on depth-masked frame instead of raw frame")
    parser.add_argument("--show_depth_text", action="store_true", help="Draw depth diagnostic text on output")
    parser.add_argument("--show_depth_side", action="store_true", help="Show RGB/nozzle output beside depth heatmap")

    # Camera / display / recording
    parser.add_argument("--camera_index", type=int, default=0, help="Camera index/source for VideoStream")
    parser.add_argument("--camera_warmup", type=float, default=1.0, help="Camera warmup time in seconds")
    parser.add_argument("-o", "--output_path", default="output_vids", help="Folder for output video if recording")
    parser.add_argument("--record", action="store_true", help="Record annotated output MP4")
    parser.add_argument("--output_fps", type=float, default=20.0, help="Output video FPS")
    parser.add_argument("--fourcc", default="mp4v", help="Video codec fourcc, e.g. mp4v or XVID")
    parser.add_argument("--display_width", type=int, default=1280, help="Resize preview window to this width; set -1 to disable")
    parser.add_argument("--window_name", default="Depth Nozzle GPIO - Press ESC", help="OpenCV window name")
    parser.add_argument("--print_every", type=int, default=10, help="Print diagnostics every N frames")

    # GPIO
    parser.add_argument("--led_pin1", type=int, default=7, help="Jetson GPIO BOARD pin for top zone (Nozzle 1)")
    parser.add_argument("--led_pin2", type=int, default=33, help="Jetson GPIO BOARD pin for middle zone (Nozzle 2)")
    parser.add_argument("--led_pin3", type=int, default=35, help="Jetson GPIO BOARD pin for bottom zone (Nozzle 3)")
    parser.add_argument("--no_gpio", action="store_true", help="Disable GPIO output, useful for testing off-Jetson")

    args = parser.parse_args()

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    if args.display_width is not None and args.display_width < 0:
        args.display_width = None

    if len(args.fourcc) != 4:
        raise ValueError("--fourcc must be exactly 4 characters, e.g. mp4v or XVID")

    return args


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    parsed_args = parse_args()
    run_live(parsed_args)