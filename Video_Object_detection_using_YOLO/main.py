from ultralytics import YOLO
import cv2


class VideoObjectDetection:
    def __init__(self, model="yolo11n.pt"):
        # Load YOLO model
        self.model = YOLO(model)
    
    def open_video(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        return cap

    def video_writer(self, first_frame, output_path, fps=30):
        # Get exact frame size from first frame
        height, width = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        return out

    def draw_boxes(self, frame, result):
        boxes = result.boxes.xyxy
        classes = result.boxes.cls
        confs = result.boxes.conf

        for box, c, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = map(int, box)
            class_name = self.model.names[int(c)]
            label = f"{class_name} {conf:.2f}"

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame

    def process_video(self, input_video, output_video):
        cap = self.open_video(input_video)

        # Read first frame to get correct size
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read video")
            return

        # Get FPS from video if available
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # fallback

        out = self.video_writer(frame, output_video, fps)

        # Process first frame
        results = self.model(frame,conf = 0.2)
        frame = self.draw_boxes(frame, results[0])
        out.write(frame)

        # Process remaining frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame)
            frame = self.draw_boxes(frame, results[0])
            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Processing complete! Video saved at: {output_video}")


if __name__ == "__main__":
    input_path = "videos/video2.mp4"
    output_path = "result_videos/output_test.mp4"

    detector = VideoObjectDetection()
    detector.process_video(input_path, output_path)  