from ultralytics import YOLO
import cv2
import numpy as np


class VideoObjectSegmentation:

    def __init__(self, model="yolo11n-seg.pt"):
        self.model = YOLO(model)

    def draw_segmentation(self, frame, result):

        if result.masks is None:
            return frame

        overlay = frame.copy()

        for mask_xy, box, cls, conf in zip(
                result.masks.xy,
                result.boxes.xyxy,
                result.boxes.cls,
                result.boxes.conf,
        ):

            polygon = mask_xy.astype(np.int32).reshape((-1, 1, 2))

            cv2.fillPoly(overlay, [polygon], (0, 255, 0))

            cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

            x1, y1, x2, y2 = map(int, box)
            label = f"{self.model.names[int(cls)]} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        return frame

    def process_video(self, input_path, output_path):

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print("Error opening video")
            return

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

            result = self.model.track(
                frame,
                persist=True,
                conf=0.2,
                classes=[0, 2],
                iou=0.5,
            )[0]

            print(f"Detected {len(result.boxes)} objects in current frame")

            frame = self.draw_segmentation(frame, result)
            out.write(frame)

        cap.release()
        out.release()
        print("Video saved at:", output_path)


if __name__ == "__main__":

    input_video  = "videos/test.mp4"
    output_video = "result_videos/output_segmentation.mp4"

    input_video2  = "videos/video2.mp4"
    output_video2 = "result_videos/output_segmentation2.mp4"

    segmentor = VideoObjectSegmentation()

    segmentor.process_video(input_video,  output_video)
    segmentor.process_video(input_video2, output_video2)