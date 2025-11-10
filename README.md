# Tennis Tracking System

Pure computer vision approach for tennis player and ball tracking from broadcast footage to bird's eye view. No machine learning except for court corner detection and benchmarking against SOTA .

## Pipeline

```
TV Broadcast → Frame Classification → Court Detection → Player/Ball Tracking → Perspective Warp → Bird's Eye View
```

### 1. Frame Classification (frame_predict/)
- KNN-based classifier identifies valid court frames
- Filters out replays, cuts, and non-gameplay footage
- Features: line orientation, white pixel density, texture sharpness

### 2. Court Corner Detection (utils/detect_court_corners.py)
- Baseline-anchored approach using Canny edge detection
- Detects bottom baseline first, then uses court geometry for top corners
- Creates inflated court boundary mask for tracking region

### 3. Player Tracking (trackers/player_tracker.py)
- MOG2 background subtraction (history=500, varThreshold=50)
- Morphological operations: open, dilate, close with elliptical kernels
- Kalman filtering for smooth trajectory prediction
- Feet position tracking (bottom center of bounding box)
- IOU-based track matching with centroid distance

### 4. Ball Tracking (trackers/ball_tracker.py)
- MOG2 background subtraction with frame differencing
- Contrast enhancement and morphological refinement
- Circularity and compactness shape validation
- Player region masking to avoid false positives
- Kalman filtering (2 frame init, 3 frame prediction)
- Court boundary constraints to filter outliers

### 5. Perspective Transformation (utils/corner_detector.py)
- Perspective warp using detected corners
- Maps player/ball positions to bird's eye coordinates
- Vertical padding for out-of-bounds tracking

### 6. Visualization (utils/visualisation.py)
- Real-time matplotlib plotting (dual view)
- Player trails and bounding boxes
- Ball position tracking
- Court overlay on warped view

## Testing

YOLOv8n models trained for comparison:
- Player tracking: person class detection
- Ball tracking: custom trained model

Located in trackers/player_tracker_yolo.py and trackers/ball_tracker_yolo.py

## Usage

```bash
python main.py
```

Outputs:
- tracking_unified.csv: Combined tracking data
- ball_tracking_raw.csv: Raw ball coordinates (MOG2 vs YOLO)
- Real-time visualization window

## Implementation

Main components:
- main.py: Processing pipeline
- trackers/: MOG2 and YOLO tracking implementations
- utils/: Corner detection, warping, visualization
- frame_predict/: Court frame classification
- court_detect/: Neural network corner detection (alternative)
