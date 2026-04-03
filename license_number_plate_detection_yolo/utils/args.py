import argparse


def parse_train_args():
    parser = argparse.ArgumentParser(description="Train YOLOv11 on a custom dataset")

    parser.add_argument("--WANDB_API_KEY", type=str, required=True, help="Weights and Biases API key")
    parser.add_argument("--weights", type=str, default="weights/yolo11n.pt", help="Path to model weights (default is weights/yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default="cuda", help="Training device: cuda or cpu")
    parser.add_argument("--project-name", type=str, default="Object_detection", help="YOLO training project folder")
    parser.add_argument("--run-name", type=str, default="yolo11n_gpu", help="YOLO training run name")

    return parser.parse_args()


def parse_predict_args():
    parser = argparse.ArgumentParser(description="YOLOv11 Inference")

    parser.add_argument("--image", type=str, default="https://miro.medium.com/v2/resize:fit:1400/1*qre-gAVNTuazaUPvNw2w-Q.jpeg", help="Image source: URL or local file path")
    parser.add_argument("--image-output", type=str, default="data/test_images", help="Directory to save downloaded input image")
    parser.add_argument("--weights", type=str, default="models/best.pt", help="Path to model weights (default is models/best.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save", type=str, default="data/predicted_images", help="Directory to save predicted images")

    return parser.parse_args()