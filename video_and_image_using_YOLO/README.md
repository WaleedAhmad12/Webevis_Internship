# Video Processing with YOLO11

Process videos using YOLO11 for detection, tracking, segmentation, and pose estimation.

---

## Project Structure

```
video_and_image_using_YOLO/
├── data/
│   ├── videos/               # Input videos
│   └── result_videos/        # Output processed videos
├── model/
│   └── image_processor.py    # YOLO model wrapper
├── utils/
│   └── image_utils.py        # Drawing utilities
├── weights/                  # YOLO weight files (auto-downloaded)
├── main.py                   # Entry point
├── requirements.txt
└── README.md
```

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/WaleedAhmad12/Webevis_Internship.git
cd Webevis_Internship/video_and_image_using_YOLO
```

**2. Create and activate virtual environment**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / Mac
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Weights

Weights are **auto-downloaded** from Google Drive on first run if not present locally.

| Mode | Weights File |
|---|---|
| detection | `weights/yolo11n.pt` |
| tracking | `weights/yolo11n.pt` |
| segmentation | `weights/yolo11n-seg.pt` |
| pose_estimation | `weights/yolo11n-pose.pt` |

You can also place weights manually in the `weights/` folder to skip the download.

---

## Usage

### Single video
```bash
python main.py \
  --input  data/videos/video1.mp4 \
  --output data/result_videos \
  --mode   tracking \
  --weights weights/yolo11n.pt
```

### Multiple videos
```bash
python main.py \
  --input  data/videos/video1.mp4 data/videos/video2.mp4 \
  --output data/result_videos \
  --mode   detection
```

### All arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--input` | Yes | — | One or more input video paths |
| `--output` | Yes | — | Output folder to save results |
| `--mode` | No | `detection` | Processing mode (see below) |
| `--weights` | No | `weights/yolo11n.pt` | Path to YOLO weights file |

### Modes

| Mode | Description | Required Weights |
|---|---|---|
| `detection` | Detects and labels objects | `yolo11n.pt` |
| `tracking` | Tracks objects with unique IDs | `yolo11n.pt` |
| `segmentation` | Instance segmentation with masks | `yolo11n-seg.pt` |
| `pose_estimation` | Human pose with skeleton keypoints | `yolo11n-pose.pt` |

---

## Examples

```bash
# Detection
python main.py --input data/videos/video1.mp4 --output data/result_videos --mode detection

# Tracking
python main.py --input data/videos/video1.mp4 --output data/result_videos --mode tracking

# Segmentation
python main.py --input data/videos/video1.mp4 --output data/result_videos --mode segmentation --weights weights/yolo11n-seg.pt

# Pose Estimation
python main.py --input data/videos/video1.mp4 --output data/result_videos --mode pose_estimation --weights weights/yolo11n-pose.pt
```

---

## Requirements

- Python 3.8+
gdown==5.2.1
numpy==2.4.3
opencv-python==4.13.0.92
ultralytics==8.4.21
- See `requirements.txt` for full list
