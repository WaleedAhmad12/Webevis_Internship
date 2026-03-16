import argparse
import cv2
from model.image_processor import ImageProcessor
from utils.image_utils import draw
import os
import gdown

WeightsGdriveIDs = {
    "yolo11n.pt": "1cjris_0oi7Q2dOUzywskPupeGmihbau4",
    "yolo11n-seg.pt": "1tj3nqdWQYZDE8yoWUQ9pQjDDrDfT0ajN",
    "yolo11n-pose.pt": "1VXfVoVW_dHU-mAEk-CR9EaZ84cOp81I2"

}


def check_weights(weights_path):
    print(f"Checking for weights at: {weights_path}")
    if os.path.exists(weights_path):
        print(f"Weights found at {weights_path}.")
        return 
    
    filename = os.path.basename(weights_path)
    file_id = WeightsGdriveIDs.get(filename)

    if file_id is None:
        print(f"No download link available for {filename}. Please provide the weights manually.")

    print(f"Downloading {filename} from Google Drive...")
    gdown.download(id=file_id, output=weights_path, quiet=False)
    print(f"Weights downloaded successfully to: {weights_path}")
    return

def process_video(input_path, output_path, mode, weights_path):
    
    check_weights(weights_path)

    print("Initializing model...")

    model = ImageProcessor(weights_path)
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Error opening video")
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    out = None

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if out is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        result = model.process_image(frame, mode)

        frame = draw(frame, result, mode)
        out.write(frame)

    cap.release()
    out.release()
    print("Video saved at:", output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process a video with YOLOv11")

    parser.add_argument("--input", type = str, required=True, nargs='+',help="Path to input video")
    parser.add_argument("--output",type = str, required = True,help= "Path to save output video")
    parser.add_argument("--mode",type = str,default="detection",choices=["detection","tracking","segmentation","pose_estimation"],help="Processing mode")
    parser.add_argument("--weights",type=str,default="weights/yolo11n.pt",help="Path to model weights (default is weights/yolo11n.pt)")

    args = parser.parse_args()

    print("Starting video processing...")

   # process_video(args.input, args.output, args.mode, weights_path=args.weights)

    print(f"🚀 Processing {len(args.input)} video(s)...\n")

    for i, input_path in enumerate(args.input):       

        filename    = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(args.output, f"{filename}_result.mp4")

        print(f"── Video {i+1}/{len(args.input)}: {input_path} → {output_path}")
        process_video(input_path, output_path, args.mode, args.weights)
