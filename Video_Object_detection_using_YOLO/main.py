from ultralytics import YOLO
import cv2


class VideoObjectDetection:

    def __init__(self, model="yolo11n.pt"):
        self.model = YOLO(model)

    def draw_boxes(self, frame, result):

        for box, cls, conf in zip(result.boxes.xyxy,
                                  result.boxes.cls,
                                  result.boxes.conf):

            x1, y1, x2, y2 = map(int, box)
            label = f"{self.model.names[int(cls)]} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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

        
            result = self.model(frame, conf=0.2, classes=[0,2])[0]
            print(f"Detected {len(result.boxes)} objects in current frame")
            
            frame = self.draw_boxes(frame, result)

            out.write(frame)

        cap.release()
        out.release()
        print("Video saved at:", output_path)


if __name__ == "__main__":

    input_video = "videos/test.mp4"
    output_video = "result_videos/output_test.mp4"


    input_video2 = "videos/video2.mp4"
    output_video2 = "result_videos/output_test2.mp4"

    detector = VideoObjectDetection()

    detector.process_video(input_video, output_video)
    detector.process_video(input_video2, output_video2)
