import os
import torch
import wandb
import gdown
from roboflow import Roboflow


WeightsGdriveIDs = {
    "yolo11n.pt": "1cjris_0oi7Q2dOUzywskPupeGmihbau4"
}


def dataset_download():
    print("dataset_download_started")

    rf = Roboflow(api_key="nKR7maxkLCNkzO6PCUa0")
    project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
    version = project.version(13)
    dataset = version.download("yolov11")

    print("Dataset downloaded to:", dataset.location)

    yaml_path = os.path.join(dataset.location, "data.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"[dataset] data.yaml not found at expected path: {yaml_path}\n"
            f"Contents of {dataset.location}: {os.listdir(dataset.location)}"
        )

    print(f"[dataset] data.yaml found at: {yaml_path}")
    return yaml_path


def weights_biases_login(wandb_api_key: str):
    wandb.login(key=wandb_api_key)
    print("Weights and biases logged in.")


def weights_check(weights_filename: str):
    print(f"Checking for weights at: {weights_filename}")
    if os.path.exists(weights_filename):
        print(f"Weights found at {weights_filename}.")
        return weights_filename

    filename = os.path.basename(weights_filename)
    file_id = WeightsGdriveIDs.get(filename)

    if file_id is None:
        print(f"No download link available for {filename}. Please provide the weights manually.")
        return weights_filename

    print(f"Downloading {filename} from Google Drive...")
    gdown.download(id=file_id, output=weights_filename, quiet=False)
    print(f"Weights downloaded successfully to: {weights_filename}")
    return weights_filename


def device_check(device: str) -> str:
    print(f"[device] CUDA available : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"[device] GPU : {torch.cuda.get_device_name(0)}")
    elif device == "cuda":
        print("[device] WARNING: CUDA requested but not available – falling back to CPU.")
        return "cpu"

    return device


WeightsPredictGdriveIDs = {
    "best.pt": "1NZWR1O-HGgWswTAY9qZ3kuDG2YqyD7yG"
}


def get_model(weights_path: str):
    if os.path.exists(weights_path):
        print(f"Weights found at {weights_path}.")
        return weights_path

    filename = os.path.basename(weights_path)

    if filename in WeightsPredictGdriveIDs:
        file_id = WeightsPredictGdriveIDs[filename]
        print(f"Downloading {filename} from Google Drive...")
        gdown.download(id=file_id, output=weights_path)
        print(f"Weights downloaded successfully to: {weights_path}")
        return weights_path

    raise FileNotFoundError(f"Model '{filename}' not found locally and no GDrive ID available.")


def download_image(image_source: str, output_dir: str):
    from datetime import datetime
    import requests

    timestamp = datetime.now().strftime("%H%M%S")

    if image_source.startswith("http://") or image_source.startswith("https://"):
        print(f"Downloading image from URL: {image_source}")
        response = requests.get(image_source)
        response.raise_for_status()

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"input_{timestamp}.jpg")

        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Image saved to: {output_path}")
    else:
        output_path = image_source
        print(f"Using local image: {output_path}")

    return output_path


def run_prediction(model_path: str, image_path: str, imgsz: int, conf: float, save: str):
    from datetime import datetime
    from ultralytics import YOLO

    model = YOLO(model_path)
    results = model(image_path, imgsz=imgsz, conf=conf)

    os.makedirs(save, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S")

    for i, result in enumerate(results):
        output_path = os.path.join(save, f"predicted_{timestamp}_{i}.jpg")
        result.save(filename=output_path)
        print(f"Saved to: {output_path}")

    return results