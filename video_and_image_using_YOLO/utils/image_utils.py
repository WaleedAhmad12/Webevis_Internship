import cv2
import numpy as np 

def detect(frame, result):
    for box, cls, conf in zip(result.boxes.xyxy,
                                  result.boxes.cls,
                                  result.boxes.conf):

            x1, y1, x2, y2 = map(int, box)
            label = f"{result.names[int(cls)]} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame
     

def track(frame,result):

    if result.boxes.id is None:
        return frame

    #target_id = 1

    for box, cls, conf, obj_id in zip(
            result.boxes.xyxy,
            result.boxes.cls,
            result.boxes.conf,
            result.boxes.id,
    ):
        #getattr(result.boxes, 'id', [])
        #if int(obj_id) != target_id:
         #   continue  

        x1, y1, x2, y2 = map(int, box)
        label = f"{result.names[int(cls)]} {conf:.2f} ID:{int(obj_id)}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame


def segment(frame,result):
    
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
        label = f"{result.names[int(cls)]} {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    return frame


def pose_estimation(frame, result):
     
    SKELETON = [
        (0, 1), (0, 2),           # nose -> eyes
        (1, 3), (2, 4),           # eyes -> ears
        (5, 6),                   # shoulders
        (5, 7), (7, 9),           # left arm
        (6, 8), (8, 10),          # right arm
        (5, 11), (6, 12),         # torso
        (11, 12),                 # hips
        (11, 13), (13, 15),       # left leg
        (12, 14), (14, 16),       # right leg
    ]

    KEYPOINT_COLOR = (0, 255, 255)    # yellow
    SKELETON_COLOR = (0, 255, 0)      # green
    BBOX_COLOR     = (255, 128, 0)    # orange

    if result.keypoints is None:
        return frame

    
    keypoints_data = result.keypoints.xy.cpu().numpy()       # (N, 17, 2)
    confidences    = result.keypoints.conf.cpu().numpy()     # (N, 17)

    for person_kps, person_conf, box, conf in zip(
            keypoints_data,
            confidences,
            result.boxes.xyxy,
            result.boxes.conf,
    ):
            # --- bounding box ---
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), BBOX_COLOR, 2)

        label = f"person {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BBOX_COLOR, 2)

        # --- skeleton limbs ---
        for pt_a, pt_b in SKELETON:
            if person_conf[pt_a] < 0.5 or person_conf[pt_b] < 0.5:
                continue

            x_a, y_a = int(person_kps[pt_a][0]), int(person_kps[pt_a][1])
            x_b, y_b = int(person_kps[pt_b][0]), int(person_kps[pt_b][1])

            cv2.line(frame, (x_a, y_a), (x_b, y_b), SKELETON_COLOR, 2)

        # --- keypoint dots ---
        for kp_x, kp_y, kp_conf in zip(
            person_kps[:, 0], person_kps[:, 1], person_conf
        ):
            if kp_conf < 0.5:
                continue
            cv2.circle(frame, (int(kp_x), int(kp_y)), 4, KEYPOINT_COLOR, -1)

    return frame


def draw(frame, result, mode):

    if mode == "detection":
        return detect(frame,result)

    elif mode == "tracking":
        return track(frame,result)

    elif mode == "segmentation":
        return segment(frame,result)

    elif mode == "pose_estimation":
        return pose_estimation(frame,result)

    else:
        print(f"Unknown mode: '{mode}'. Choose from: detection, tracking, segmentation, pose_estimation")
   
    return frame
