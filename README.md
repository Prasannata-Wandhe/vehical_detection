---
title: Vehicle Distance Measurement System
emoji: 🚗
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
---

# Vehicle Distance Measurement System

## 1. Introduction

This project implements a real-time vehicle distance measurement system designed for in-vehicle dashcam applications. Using YOLOv12 object detection and monocular depth estimation through perspective geometry, the system calculates distances to surrounding vehicles with region-specific warning systems. The system monitors three ROI lanes (LEFT, MAIN, RIGHT) and automatically blurs vehicle plates for GDPR compliance, providing an adaptable safety assistance system for autonomous vehicles and advanced driver monitoring.

**Core Features:**
- Real-time multi-vehicle detection using YOLOv12
- Accurate distance calculation through perspective projection
- Three-zone ROI monitoring (LEFT, MAIN, RIGHT lanes)
- Adaptable warning thresholds per lane
- Automatic vehicle plate detection and blurring
- Color-coded distance visualization
- Live safety indicators
- Annotated video recording

## 2. Methodology / Approach

The system employs a multi-component architecture combining object detection, perspective geometry, and region-based analysis:

**Object Detection:** YOLOv12 detects vehicles (cars, motorcycles, buses, trucks) and vehicle plates in real-time with high confidence.

**Distance Estimation:** Uses monocular depth estimation via perspective projection, calculating distance from bounding box height using calibrated focal length and known vehicle dimensions.

**ROI Zone Analysis:** Frame divided into three trapezoidal regions (LEFT, MAIN, RIGHT) with separate warning thresholds and display limits for context-aware alerting.

**Privacy Protection:** Automatic vehicle plate detection and Gaussian blurring for regulatory compliance.

**Adaptive Visualization:** Color-coded distance labels (red for warnings, green for safe) with zone-specific thresholds.

### 2.1 System Architecture

```
[Dashcam Video Input]
    ↓
[YOLOv12 Vehicle Detection]
    ↓
[ROI Zone Classification]
    ↓
[Distance Calculation] → [Perspective Correction]
    ↓
[Vehicle Plate Detection] → [Blurring]
    ↓
[Warning Assessment] → [Color Coding]
    ↓
[Visualization & Output]
    ↓
[Annotated Video Output]
```

### 2.2 Processing Pipeline

1. Capture frame from dashcam video
2. Run YOLOv12 detection on full frame
3. Filter detections to vehicle classes (2, 3, 5, 7)
4. Classify vehicle center into ROI zone (LEFT/MAIN/RIGHT)
5. Calculate distance using perspective projection formula
6. Apply perspective distortion correction based on offset
7. Detect vehicle plates within vehicle regions
8. Apply Gaussian blur to vehicle plates
9. Evaluate distance against zone-specific warning threshold
10. Color-code distance label (RED/GREEN) accordingly
11. Overlay distance text and warning indicators
12. Record annotated frame to output video

## 3. Mathematical Framework

### 3.1 Perspective Projection Distance Calculation

The fundamental equation relating observed bounding box height to actual distance:

$$d = \frac{h_{\text{real}} \cdot f}{h_{\text{image}}}$$

where:
- $d$ = distance from camera to vehicle (meters)
- $h_{\text{real}}$ = actual vehicle height (meters)
- $f$ = camera focal length (pixels)
- $h_{\text{image}}$ = bounding box height in pixels

### 3.2 ROI Zone Definition

Three trapezoidal regions for multi-lane monitoring (2042×1148 resolution):

$$\mathbf{ROI}_{\text{LEFT}} = \{(x, y) : (x, y) \in \text{Polygon}([[240, 600], [925, 550], [312, 1100], [100, 1100]])\}$$

$$\mathbf{ROI}_{\text{MAIN}} = \{(x, y) : (x, y) \in \text{Polygon}([[925, 550], [1025, 550], [1712, 1100], [312, 1100]])\}$$

$$\mathbf{ROI}_{\text{RIGHT}} = \{(x, y) : (x, y) \in \text{Polygon}([[1025, 550], [1802, 600], [1942, 1100], [1712, 1100]])\}$$

### 3.3 Vehicle Height Classification

Reference heights for distance calculation by vehicle type:

$$H_{\text{vehicle}} = \begin{cases}
1.55 \text{ m} & \text{if class ID} = 2 \text{ (Car)} \\
1.2 \text{ m} & \text{if class ID} = 3 \text{ (Motorcycle)} \\
3.0 \text{ m} & \text{if class ID} = 5 \text{ (Bus)} \\
2.5 \text{ m} & \text{if class ID} = 7 \text{ (Truck)}
\end{cases}$$

### 3.4 Perspective Distortion Correction

Correction factor for off-center vehicles due to perspective distortion:

$$d_{\text{corrected}} = d \cdot (1 + \alpha \cdot \delta)$$

where:
- $\alpha = 0.0001$ (displacement coefficient)
- $\delta$ = Euclidean distance from optical center to vehicle center

$$\delta = \sqrt{(x_{\text{vehicle}} - x_{\text{center}})^2 + (y_{\text{vehicle}} - y_{\text{center}})^2}$$

## 4. Requirements

```txt
opencv-python>=4.5.0
numpy>=1.21.0
ultralytics>=8.0.0
flask>=3.0.0
```

## 5. Installation & Configuration

Install the required packages:

```bash
pip install -r requirements.txt
```

### 5.2 Project Structure

```
Vehicle-Distance-Measurement-System/
├─ Vehicle-Distance-Measurement-System.py
├─ web_app.py
├─ templates/
│  └─ index.html
├─ README.md
├─ requirements.txt
└─ LICENSE
```

### 5.3 Required Files

- **Vehicle YOLO Model:** `yolo12x.pt` (downloaded by Ultralytics when needed)
- **Plate YOLO Model:** `vehicle-plate.pt` (optional; used for plate blurring)
- **Input Image:** JPG, PNG, or WEBP for the web application
- **Input Video:** Dashcam video in MP4, MOV, or AVI format for video processing

## 6. Usage / How to Run

### 6.1 Web Application

Start the local upload interface:

```bash
python web_app.py
```

Open the following URL in a browser:

```text
http://127.0.0.1:5000
```

Upload a road image to receive:

- An annotated image with vehicle bounding boxes
- Vehicle type and detection confidence
- ROI zone: `LEFT`, `MAIN`, or `RIGHT`
- Estimated distance in meters
- Warning status

The web application uses `yolo12n.pt` by default when that model is available. To use another vehicle model:

```bash
set VEHICLE_MODEL=yolo12x.pt
python web_app.py
```

On PowerShell, use `$env:VEHICLE_MODEL = "yolo12x.pt"` instead.

### 6.2 Video Processing

Run the original video pipeline with configurable input and output paths:

```bash
python Vehicle-Distance-Measurement-System.py --input dashcam_video.mov --output Vehicle-Distance-Measurement.mp4
```

For testing without the optional plate model:

```bash
python Vehicle-Distance-Measurement-System.py --input dashcam_video.mov --output Vehicle-Distance-Measurement.mp4 --vehicle-model yolo12n.pt --no-plate-blur
```

### 6.3 Deploying the Web Application to Vercel

The repository includes `vercel.json` and `api/index.py` for Vercel's Python serverless runtime.

1. Import this GitHub repository into Vercel.
2. Leave the project root as the repository root.
3. Select **Other** as the framework preset if Vercel asks for one.
4. Add the vehicle model file `yolo12n.pt` to the repository, or provide another model through the `VEHICLE_MODEL` environment variable.
5. Deploy and open the Vercel-provided URL.

The deployed app accepts image uploads at `/` and analyzes them through `/api/analyze`. Vercel's serverless functions are not suitable for long-running video processing; use the local video command for that workflow.

### 6.4 Deploying to Hugging Face Spaces

This repository includes a `Dockerfile` configured for Hugging Face Spaces. Create a new **Docker Space**, then upload or push this repository to it. The Space serves the upload application on port `7860`.

The default web model is `yolo12n.pt`; Ultralytics downloads it the first time an image is analyzed. For a private or custom model, set the Space variable `VEHICLE_MODEL` and make the model available to the container.

### 6.5 Configuration

Update the script parameters for your specific setup:

```python
# Video input/output
video_capture = cv2.VideoCapture("dashcam_video.mov")
output_file = 'Vehicle-Distance-Measurement.mp4'

# ROI Zones (modify for different camera angles)
ROI_ZONES = {
    'LEFT': [[240, 600], [925, 550], [312, 1100], [100, 1100]],
    'MAIN': [[925, 550], [1025, 550], [1712, 1100], [312, 1100]],
    'RIGHT': [[1025, 550], [1802, 600], [1942, 1100], [1712, 1100]]
}

# Optical centers per zone (camera calibration)
OPTICAL_CENTERS = {
    'LEFT': (500, 800),
    'MAIN': (1025, 900),
    'RIGHT': (1550, 800)
}

# Warning distances per zone (meters)
WARNING_DISTANCES = {
    'LEFT': 1.0,
    'MAIN': 2.0,
    'RIGHT': 1.0
}

# Camera parameters
FOCAL_LENGTH = 500  # pixels
VEHICLE_CONFIDENCE = 0.7
VEHICLE_PLATE_CONFIDENCE = 0.475
```

### 6.6 Controls

- Press `q` to quit the application

### 6.7 Output

The processed video is saved as:
```
Vehicle-Distance-Measurement.mp4
```

## 7. Application / Results

### 7.1 Input Video

[![Vehicle Plate Blurring](https://github.com/kemalkilicaslan/Vehicle-Distance-Measurement-System/blob/main/Vehicle-License-Plate-Blurring.webp)](https://www.youtube.com/watch?v=x1u2iy4csSI)

### 7.2 Vehicle Distance Measurement in Region of Interest

[![Vehicle Distance Measurement in Region of Interest](https://github.com/kemalkilicaslan/Vehicle-Distance-Measurement-System/blob/main/Vehicle-Distance-Measurement-System-in-ROI.webp)](https://www.youtube.com/watch?v=XFv5MAkfYpA)

### 7.3 Output Video

[![Vehicle Distance Measurement](https://github.com/kemalkilicaslan/Vehicle-Distance-Measurement-System/blob/main/Vehicle-Distance-Measurement.webp)](https://www.youtube.com/watch?v=1QwOh9GxzV8)

## 8. System Configuration

### 8.1 Vehicle Classes

| Class ID | Vehicle Type | Reference Height |
|----------|--------------|------------------|
| 2 | Car | 1.55 m |
| 3 | Motorcycle | 1.2 m |
| 5 | Bus | 3.0 m |
| 7 | Truck | 2.5 m |

### 8.2 Warning System Thresholds

| Lane | Warning Distance | Display Limit |
|------|------------------|---------------|
| LEFT | 1.0 m | 5.0 m |
| MAIN | 2.0 m | 15.0 m |
| RIGHT | 1.0 m | 5.0 m |

### 8.3 Color Coding

- **RED:** Vehicle distance below warning threshold OR main lane vehicles < 5m
- **GREEN:** Safe distances above threshold
- **No Label:** Distances beyond display limit for lane

### 8.4 System Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| Focal Length | 500 | pixels | Camera focal length calibration |
| Vehicle Confidence | 0.7 | - | Detection threshold for vehicles |
| Vehicle Plate Confidence | 0.475 | - | Detection threshold for plates |
| Max Display Distance | 15 | meters | Maximum distance shown in MAIN lane |
| Displacement Coefficient | 0.0001 | - | Perspective correction factor |

## 9. Tech Stack

### 9.1 Core Technologies

- **Language:** Python 3.7+
- **Computer Vision:** OpenCV 4.5+
- **Deep Learning:** Ultralytics YOLO 8.0+
- **Object Detection & Tracking:** YOLOv12
- **Numerical Computing:** NumPy 1.21+

### 9.2 Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| opencv-python | 4.5+ | Video I/O, image processing, visualization |
| ultralytics | 8.0+ | YOLOv12 vehicle and vehicle plate detection |
| numpy | 1.21+ | Array operations and geometric calculations |

### 9.3 Pre-trained Models

- **YOLOv12 (Extra Large):** `yolo12x.pt`
  - Architecture: YOLOv12 deep learning model
  - Classes: 80 COCO classes including vehicles
  - Purpose: Vehicle detection and classification

- **Vehicle Plate Detection Model:** `vehicle-plate.pt`
  - Specialized model for vehicle plate regions
  - Trained on vehicle plates
  - Purpose: Privacy protection through automated blurring

## 10. License

This project is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://github.com/kemalkilicaslan/Vehicle-Distance-Measurement-System/blob/main/LICENSE). 

## 11. References

1. Ultralytics [YOLOv12](https://docs.ultralytics.com/) Documentation.
2. OpenCV [Camera Calibration and 3D Reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) Documentation.

---

### Acknowledgments

Special thanks to the Ultralytics team for developing and maintaining the YOLO framework and YOLOv12 models. This project benefits from the OpenCV community's excellent camera calibration and computer vision tools. The perspective projection methodology is based on established pinhole camera models in computer vision literature. Sample dashcam footage used for demonstration purposes only.

---

**Note:** This system is calibrated for specific dashcam configurations. Recalibrate focal length and ROI zones when using different camera equipment. Ensure compliance with local laws regarding vehicle data collection and dashcam recording. This project is intended for research, educational, and authorized commercial applications in vehicle safety systems. Always prioritize driver safety and avoid distraction when using in-vehicle monitoring systems.