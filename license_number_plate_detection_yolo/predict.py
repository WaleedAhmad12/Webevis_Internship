from utils.args import parse_predict_args
from utils.utils import get_model, download_image, run_prediction


def main():
    args = parse_predict_args()

    image_path = download_image(args.image, args.image_output)
    model_path = get_model(args.weights)
    run_prediction(
        model_path=model_path,
        image_path=image_path,
        imgsz=args.imgsz,
        conf=args.conf,
        save=args.save,
    )


if __name__ == "__main__":
    main()