# Video Object Detection and Tracking using YOLO

This project implements object detection and tracking on video files using the YOLO (You Only Look Once) model, specifically YOLOv11 nano. It processes input videos to detect and track objects such as persons and cars, drawing bounding boxes with confidence scores and unique IDs on the output videos.

## Features

- **Object Detection**: Detects objects in video frames using YOLOv11n model.
- **Object Tracking**: Tracks detected objects across frames, assigning unique IDs.
- **Video Processing**: Processes MP4 video files and outputs annotated videos.
- **Customizable Classes**: Currently configured to detect persons (class 0) and cars (class 2).
- **Real-time Annotation**: Draws bounding boxes, labels, confidence scores, and tracking IDs on frames.

## Installation

1. **Clone or Download the Repository**:
   - Ensure you have the project files in your workspace.

2. **Install Dependencies**:
   - Install the required Python packages using pip:
     ```
     pip install -r requirements.txt
     ```

   The main dependencies include:
   - `ultralytics`: For YOLO model loading and inference.
   - `opencv-python`: For video capture, processing, and writing.
   - `torch` and `torchvision`: For PyTorch-based computations.

3. **Model Weights**:
   - The project uses `yolo11n.pt` (YOLOv11 nano) located in the `weights/` folder.
   - Ensure the weights file is present; it can be downloaded from the Ultralytics YOLO repository if needed.

## Usage

1. **Prepare Videos**:
   - Place your input video files in the `videos/` folder (e.g., `test.mp4`, `video2.mp4`).

2. **Run the Script**:
   - Execute the main script:
     ```
     python main.py
     ```
   - The script will process the specified videos and save the output in the `result_videos/` folder.

3. **Output**:
   - Annotated videos will be saved as `output_test.mp4` and `output_test2.mp4` in the `result_videos/` directory.
   - Each frame will have bounding boxes around detected objects with labels showing class name, confidence score, and tracking ID.

## Project Structure

```
Video_Object_detection_tracking_using_YOLO/
├── main.py                 # Main script for video processing
├── requirements.txt        # Python dependencies
├── weights/
│   └── yolo11n.pt         # YOLOv11 nano model weights
├── videos/                 # Input video files
│   ├── test.mp4
│   └── video2.mp4
└── result_videos/          # Output annotated videos
    ├── output_test.mp4
    └── output_test2.mp4
```

## Requirements

- **Python**: Version 3.7 or higher.
- **Hardware**: GPU recommended for faster processing (CUDA-compatible if using GPU acceleration).
- **OS**: Compatible with Windows, Linux, or macOS.

## Configuration

- **Model**: Change the model path in `VideoObjectDetection.__init__()` if using a different YOLO model.
- **Classes**: Modify the `classes` parameter in `model.track()` to detect different object classes.
- **Confidence Threshold**: Adjust `conf` in `model.track()` for detection sensitivity.
- **IOU Threshold**: Tune `iou` for tracking overlap sensitivity.


