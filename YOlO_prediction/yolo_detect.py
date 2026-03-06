import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model(["https://ultralytics.com/images/bus.jpg", 
                 "https://ultralytics.com/images/zidane.jpg"])

for i_img, result in enumerate(results): 

    image = result.orig_img 

    xyxy   = result.boxes.xyxy
    xyxyn  = result.boxes.xyxyn
    xywh   = result.boxes.xywh
    xywhn  = result.boxes.xywhn

    cls    = result.boxes.cls
    confs  = result.boxes.conf

    print(f"--- Image {i_img + 1} ---")
    print(f"Number of objects detected: {len(xyxy)}\n")

    for i, (box_xyxy, box_xyxyn, box_xywh, box_xywhn, c, conf) in enumerate(
        zip(xyxy, xyxyn, xywh, xywhn, cls, confs)
    ):
        class_name = model.names[int(c)]

        print(f"Object {i+1}: {class_name} (confidence: {conf:.2f})")

        print(f"  xyxy   (absolute px) : x1={box_xyxy[0]:.1f},  y1={box_xyxy[1]:.1f},  x2={box_xyxy[2]:.1f},  y2={box_xyxy[3]:.1f}")
        print(f"  xyxyn  (normalized)  : x1={box_xyxyn[0]:.3f}, y1={box_xyxyn[1]:.3f}, x2={box_xyxyn[2]:.3f}, y2={box_xyxyn[3]:.3f}")
        print(f"  xywh   (absolute px) : cx={box_xywh[0]:.1f},  cy={box_xywh[1]:.1f},  w={box_xywh[2]:.1f},   h={box_xywh[3]:.1f}")
        print(f"  xywhn  (normalized)  : cx={box_xywhn[0]:.3f}, cy={box_xywhn[1]:.3f}, w={box_xywhn[2]:.3f},  h={box_xywhn[3]:.3f}")

        print()

        x1, y1, x2, y2 = int(box_xyxy[0]), int(box_xyxy[1]), int(box_xyxy[2]), int(box_xyxy[3])
        label = f"{class_name} {conf:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(f"output_image{i_img + 1}.jpg", image)
    cv2.imshow(f"Detections - Image {i_img + 1}", image)
    cv2.waitKey(0)