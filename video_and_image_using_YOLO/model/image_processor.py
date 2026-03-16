from ultralytics import YOLO
import cv2


class ImageProcessor:

    def __init__(self, weights_path):
        self.model = YOLO(weights_path)

    def detect(self, frame):

        result = self.model(frame, conf=0.2)[0]
        print(f"Detected {len(result.boxes)} objects in current frame")            
        return result

    def track(self, frame):

        result = self.model.track(
                frame,
                persist=True,
                conf=0.2,
                iou=0.5,
            )[0]
      
        print(f"Detected {len(result.boxes)} objects in current frame")
        return result

    def segment(self, frame):

        result = self.model.track(
                        frame,
                        persist=True,
                        conf=0.2,
                        iou=0.5,
                    )[0]
            
        print(f"Detected {len(result.boxes)} objects in current frame")
        return result

    def pose_estimation(self, frame):
        
        result = self.model.track(
            frame,
            persist=True,
            conf=0.2,
            classes=[0],       
            iou=0.5,
        )[0]

        print(f"Detected {len(result.boxes)} persons in current frame")
        return result

    def process_image(self, frame, mode):

        if mode == "detection":
            return self.detect(frame)

        elif mode == "tracking":
            return self.track(frame)

        elif mode == "segmentation":
            return self.segment(frame)

        elif mode == "pose_estimation":
            return self.pose_estimation(frame)

        else:
            print(f"Unknown mode: '{mode}'. Choose from: detection, tracking, segmentation, pose_estimation")
            return None