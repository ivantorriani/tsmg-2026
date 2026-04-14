# -*- coding: utf-8 -*-
"""
Updated on July 30, 2025
Author: Fahim
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from datetime import datetime
import time
import sys
import os
import Jetson.GPIO as GPIO
import time
import curses

# ----------------------------
# Get input file from command line
# ----------------------------
if len(sys.argv) < 2:
    print("Usage: python inference_yolov8.py <input_video_path>")
    sys.exit(1)

input_video_path = sys.argv[1]

if not os.path.isfile(input_video_path):
    print(f"Error: File '{input_video_path}' does not exist.")
    sys.exit(1)

# Setup GPIO pins

led_pin1 = 7
led_pin2 = 33

GPIO.setmode(GPIO.BOARD)
GPIO.setup(led_pin1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(led_pin2, GPIO.OUT, initial=GPIO.LOW)

# Load YOLOv8 model
model = YOLO('model0.pt')

# Open video capture
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Error: Could not open video {input_video_path}")
    sys.exit(1)

# Get frame size from input video
ret, frame = cap.read()
if not ret:
    print("Error: Failed to read the first frame of the video.")
    sys.exit(1)

frame_height, frame_width = frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)

# Prepare video writer
input_basename = os.path.splitext(os.path.basename(input_video_path))[0]
timestamp = str(int(time.time()))
output_filename = f"output_vids/{input_basename}_{timestamp}.mp4"

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

# Pre-compute region edges (three equal horizontal bands)
y_edges = [0, int(round(frame_height / 3)), int(round(2 * frame_height / 3)), frame_height]
nozzle_base_labels = ["Nozzle 1", "Nozzle 2", "Nozzle 3"]

# Process video
frameid = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind to the first frame

cv2.namedWindow('PreviewWindow', cv2.WINDOW_NORMAL)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_boxes = frame.copy()
    h, w, _ = frame.shape

    # Run YOLO on the raw frame
    results = model.predict(
        source = frame,
        conf = 0.5, 
        classes=[0],
        device = 0 if torch.cuda.is_available() else "cpu", 
        half = torch.cuda.is_available(),
        verbose=False
    )

    # ----- Build a union mask of all class-0 detections -----
    det_mask = np.zeros((h, w), dtype=np.uint8)

    for result in results:
        for score, cls, bbox in zip(result.boxes.conf, result.boxes.cls, result.boxes.xyxy):
            if cls.item() != 0.0:
                continue

            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = map(int, [x1.item(), y1.item(), x2.item(), y2.item()])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            cv2.rectangle(det_mask, (x1, y1), (x2 - 1, y2 - 1), 1, thickness=-1)

            # Draw detection in blue (optional)
            label = f"Canopy: {score.item() * 100:.2f}%"
            cv2.rectangle(img_boxes, (x1, y1), (x2, y2), (255, 0, 0), 4)
            cv2.putText(img_boxes, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)

    # ----- For each region, compute coverage ratio -----
    region_active = [False, False, False]

    
    for i in range(3):
        y1_area = y_edges[i]
        y2_area = y_edges[i + 1]
        region_h = max(0, y2_area - y1_area)
        if region_h == 0:
            continue

        region_area = region_h * w
        covered_pixels = int(det_mask[y1_area:y2_area, :].sum())
        coverage_ratio = covered_pixels / region_area
        region_active[i] = coverage_ratio > 0.25  # >25% region covered

    #assign each region 
    top_active = region_active[0]

    middle_active = region_active[1]

    bottom_active = region_active[2]

    #send signals to led lights

    if top_active:
        #Turn both led lights on if top is active
        GPIO.output(led_pin1, GPIO.HIGH)
        GPIO.output(led_pin2, GPIO.HIGH)
    if middle_active:
        #Turn one color on if middle is active
        GPIO.output(led_pin1, GPIO.HIGH)
    if bottom_active:
        #Turn one color on if bottom is active
        GPIO.output(led_pin2, GPIO.HIGH)
    if (not middle_active):
        GPIO.output(led_pin1, GPIO.LOW)
    if (not bottom_active): 
        GPIO.output(led_pin2, GPIO.LOW)
    


    # ----- Draw shaded regions with Active/Inactive labels -----
    for i in range(3):
        y1_draw = y_edges[i]
        y2_draw = y_edges[i + 1] - 1
        active = region_active[i]

        if active:
            # Make overlay for transparency shading
            overlay = img_boxes.copy()
            cv2.rectangle(overlay, (0, y1_draw), (w - 1, y2_draw), (0, 255, 0), -1)
            # Blend overlay with base image
            alpha = 0.3  # transparency factor
            cv2.addWeighted(overlay, alpha, img_boxes, 1 - alpha, 0, img_boxes)

        # Rectangle border (green if active, red if inactive)
        color = (0, 255, 0) if active else (0, 0, 255)
        status_text = "Active" if active else "Inactive"

        cv2.rectangle(img_boxes, (0, y1_draw), (w - 1, y2_draw), color, 3)
        cv2.putText(img_boxes, f"{nozzle_base_labels[i]} {status_text}",
                    (10, max(30, y1_draw + 40)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    out.write(img_boxes)

    #Formatting Screen

    display = cv2.resize(img_boxes, (1280,1280))
    cv2.imshow('PreviewWindow', display)

    out.write(img_boxes)

    frameid += 1

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Output saved to {output_filename}")
