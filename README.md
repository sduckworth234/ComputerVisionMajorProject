# Tennis Tracking System
**Computer Vision Project - Pure CV (No ML)**

## Overview
Real-time tennis player and ball tracking system using computer vision. Takes video input, detects court boundaries, tracks players and ball, and provides bird's-eye visualization with position tracking.

---

## Pipeline

```
Video Input → Court Corner Detection → Masking → Player & Ball Tracking → Warping → Visualization
```

### 1. **Video Input** (`utils/data_loader.py`)
- Loads video frames (MP4)
- Supports single frame or continuous stream
- Frame extraction and preprocessing

### 2. **Court Corner Detection** (`utils/corner_detector.py`)
- Detects 4 court corners using edge detection and quadrant search
- Maps corners from cropped region to full image coordinates
- Creates court mask for focused tracking (excludes crowd/background)

### 3. **Player Tracking** (`trackers/player_tracker.py`)
- **MOG2 Background Subtraction** (history=300, varThreshold=25)
- **Aggressive Morphology**: 3x3 kernel, 3 open + 7x7 kernel, 2 close iterations
- Filters by size (750-30,000 px²) and aspect ratio (0.75-10)
- Centroid tracking with 150px max distance threshold
- Maintains player IDs across frames

### 4. **Ball Tracking** (`trackers/ball_tracker.py`)
- **MOG2 Background Subtraction** (history=300, varThreshold=25)
- **Lightweight Morphology**: 3x3 kernel, 2 open iterations (no close)
- **No sharpening, no color filtering** - pure motion detection
- Filters by size (50-500 px²) and aspect ratio (0.5-2.0)
- Excludes player regions (20px margin)
- Quality scoring: prioritizes circular blobs (~15x15px, area ~225)

### 5. **Perspective Warping** (`utils/corner_detector.py`)
- Bird's-eye view transformation using detected corners
- Vertical padding for out-of-bounds tracking
- Real-time court visualization

### 6. **Visualization** (`utils/visualisation.py`, `main.py`)
- Player bounding boxes with IDs and trails
- Ball position with single green box (no history)
- Matplotlib real-time position plot
- Original view + warped bird's-eye view

---

## Project Structure

```
CV_MajorProject/
├── main.py                    # Main pipeline orchestration
├── data/
│   ├── full_video/           # Input tennis videos
│   └── sample_frames/        # Test frames
├── utils/
│   ├── data_loader.py        # Video/frame loading
│   ├── corner_detector.py    # Court detection & warping
│   ├── visualisation.py      # Drawing & plotting
│   └── metrics.py            # Performance metrics
├── trackers/
│   ├── player_tracker.py     # Player detection & tracking
│   └── ball_tracker.py       # Ball detection & tracking
└── testing/                   # Experimental scripts
```

---

## Goal

**Build an end-to-end tennis analytics pipeline:**

1. **Input**: Raw TV stream footage (live broadcast)
2. **Segmentation**: Detect and extract court regions from multi-camera angles
3. **Tracking**: Real-time player and ball position tracking
4. **Warping**: Normalize perspective to bird's-eye view
5. **Metrics**: Calculate performance stats
   - Distance travelled per player
   - Ball speed and trajectory
   - Rally length and points won
   - Court coverage heatmaps

---

## TBD: TV Stream Video Segmentation Network

**Challenge**: Current system requires manual cropping and works with single-angle footage.

**Next Phase**: Implement deep learning segmentation to:
- **Auto-detect court regions** in raw TV broadcast footage
- **Handle multiple camera angles** (baseline, net, side views)
- **Segment court vs. crowd/background** automatically
- **Switch between camera feeds** seamlessly during rallies
- **Enable direct TV stream processing** without manual intervention

**Approach**: 
- Train semantic segmentation network (U-Net/DeepLabV3) on tennis broadcast data
- Detect court boundaries, lines, and playing surface
- Extract court ROI for corner detection pipeline
- Feed into existing tracking system

---

## Current Status
✅ Court corner detection working  
✅ Player tracking stable (MOG2 + morphology)  
✅ Ball tracking simplified (3x3x2, no color)  
✅ Bird's-eye warping functional  
✅ Real-time visualization complete  

🔲 TV stream segmentation network (TBD)  
🔲 Multi-camera angle support (TBD)  
🔲 Metrics calculation (distance, speed, etc.) (TBD)

---

## Usage

```bash
python main.py
```

Configure in `main.py`:
- `video_path`: Input video file
- `crop_region`: (top, bottom, left, right) for corner detection
- `expand_pixels`: Court boundary expansion
- `vertical_padding_ratio`: Out-of-bounds space
- `show_plot`: Enable real-time position plot

---

## Notes
- Pure computer vision (no ML required for tracking)
- Optimized for broadcast-quality tennis footage
- MOG2 parameters tuned for tennis ball speed
- Player exclusion prevents ball/player confusion
- Data folder ignored in git (large video files)