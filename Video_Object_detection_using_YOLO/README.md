# Video Object Detection with YOLO

This project detects **persons and cars** in videos using **Ultralytics YOLO** and OpenCV, and outputs a new video with bounding boxes and labels.  

---

## Features

- Detect **persons (class 0)** and **cars (class 2)** in videos  
- Adjustable **confidence threshold** for detections  
- Draws bounding boxes and class labels on each frame  
- Supports multiple video processing  

---

## Requirements

- Python 3.8+  
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)  
- OpenCV  

### Install dependencies

```bash
pip install ultralytics opencv-python
```

---

## Project Structure

```
VIDEO_OBJECT_DETECTION_USING_YOLO/
│
├─ videos/
│   ├─ test.mp4
│   └─ video2.mp4
│
├─ result_videos/
│   └─ (output videos will be saved here)
│
├─ video_detection.py   # main Python script
└─ README.md
```

---

## How to Use

1. Place your input videos in the `videos/` folder.  
2. Update the paths in the script if needed:

```python
input_video = "videos/test.mp4"
output_video = "result_videos/output_test.mp4"
```

3. Run the script:

```bash
python video_detection.py
```

4. The processed videos will be saved in `result_videos/` with **bounding boxes and class labels**.  

---

## Configuration

- **Confidence threshold:**  
  Adjust `conf` parameter in `process_video()` to filter weak detections:

```python
result = self.model(frame, conf=0.2, classes=[0,2])[0]
```

- **Classes:**  
  You can detect other objects by changing the `classes` list. COCO class IDs:  

| Object  | Class ID |
|---------|----------|
| person  | 0        |
| car     | 2        |
| bag     | 24,26,28 |
| bicycle | 1        |

- **Video output format:**  
  Currently set to `mp4v` codec (`.mp4` files). Change if needed in:

```python
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
```

---

## Notes

- Increasing **confidence threshold** reduces false positives but may miss some objects.  

```python
result = self.model(frame, conf=0.5, classes=[0,2])[0]
```

- For multiple videos, the script will process them one by one.  

---

## License

This project is open-source and free to use for learning purposes.

