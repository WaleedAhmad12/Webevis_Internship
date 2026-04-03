import torch
from ultralytics import YOLO

from utils.args import parse_train_args
from utils.utils import dataset_download, weights_biases_login, weights_check, device_check


def model_train(data_yaml: str, weights_path: str, epochs: int, batch: int, imgsz: int, device: str, project_name: str, run_name: str):
    device = device_check(device)

    model = YOLO(weights_path)
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project_name,
        name=run_name,
    )
    print("[train] Training complete!")


def main():
    args = parse_train_args()

    yaml_path = dataset_download()
    weights_biases_login(args.WANDB_API_KEY)
    weights_path = weights_check(args.weights)
    model_train(yaml_path, weights_path, args.epochs, args.batch, args.imgsz, args.device, args.project_name, args.run_name)


if __name__ == "__main__":
    main()