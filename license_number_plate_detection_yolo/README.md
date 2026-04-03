# License Number Plate Detection with YOLOv11

A YOLOv11-based system that detects and localizes vehicle license plates in images with high accuracy and speed.

## Features

- Real-time license plate detection using YOLOv11
- Model training with custom datasets from Roboflow
- Automatic model weight download from Google Drive if missing
- Easy-to-use training and prediction pipelines via argparse
- Weights & Biases (W&B) integration for training metrics

## Tech Stack

| Component | Tool |
|-----------|------|
| Model | YOLOv11n (Ultralytics) |
| Dataset | Roboflow |
| Experiment Tracking | Weights & Biases |
| Model Storage | Google Drive |
| Language | Python 3.8+ |

## Setup

### 1. Clone the repository
```bash
git clone <repo-url>
cd license_number_plate_detection_yolo
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install GPU PyTorch (Optional but recommended)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Training

To train the model with a custom dataset from Roboflow:

```bash
python train.py --WANDB_API_KEY your_wandb_key --weights weights/yolo11n.pt --epochs 20 --batch 8 --imgsz 640 --device cuda --project-name car_plate_number --run-name yolo11n_gpu
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--WANDB_API_KEY` | Required | Your Weights & Biases API key |
| `--weights` | `weights/yolo11n.pt` | Path to initial model weights |
| `--epochs` | `20` | Number of training epochs |
| `--batch` | `8` | Batch size |
| `--imgsz` | `640` | Input image size |
| `--device` | `cuda` | Training device: `cuda` or `cpu` |
| `--project-name` | `rice_leaf_disease` | W&B / YOLO project folder name |
| `--run-name` | `yolo11n_gpu` | Name for this training run |

### What happens during training
- Automatically downloads the dataset from Roboflow
- Trains YOLOv11 on the license plate detection task
- Logs training metrics to Weights & Biases
- Saves the best model weights automatically

---

## Prediction

To detect license plates in an image:

```bash
python predict.py --image your_image.jpg --image-output data/test_images --weights models/best.pt --imgsz 640 --conf 0.25 --save data/predicted_images
```

### Prediction Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--image` | Sample URL | Image path (local or URL) |
| `--image-output` | `data/test_images` | Directory to save the input image |
| `--weights` | `models/best.pt` | Path to model weights |
| `--imgsz` | `640` | Inference image size |
| `--conf` | `0.25` | Confidence threshold |
| `--save` | `data/predicted_images` | Directory to save predicted output |

### Output
- Predicted images saved to `data/predicted_images/`
- Detected license plates shown with bounding boxes and confidence scores
- The default model (`best.pt`) is automatically downloaded from Google Drive on first run

---

## Project Structure

```
license_number_plate_detection_yolo/
├── train.py                 # Training script
├── predict.py               # Prediction/inference script
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── data/
│   ├── test_images/         # Input images for inference
│   └── predicted_images/    # Output predictions
├── models/
│   └── best.pt              # Trained model weights
├── weights/                 # Base/pretrained weights
└── utils/
    ├── __init__.py
    ├── args.py              # Argument parsers
    └── utils.py             # Helper functions
```

---

## Model Performance (best.pt)

### Metrics

| Metric | Value |
|--------|-------|
| mAP@50 | 0.950 |
| mAP@50-95 | 0.682 |
| Precision | 0.945 |
| Recall | 0.928 |

### Model Info

| Property | Value |
|----------|-------|
| Parameters | 2,590,035 |
| GFLOPs | 6.441 |
| Inference Speed (PyTorch) | 9.573 ms |

### Final Losses

| Loss | Train | Val |
|------|-------|-----|
| Box Loss | 0.919 | 1.121 |
| Class Loss | 0.395 | 0.412 |
| DFL Loss | 1.020 | 1.055 |

