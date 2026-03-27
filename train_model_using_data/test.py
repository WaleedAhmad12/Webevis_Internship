from ultralytics import YOLO

model = YOLO("models/best.pt")  

results = model.predict(
    source="0", 
    conf=0.5,                
    save=True,              
    show=True                 
)


# Print detections
for r in results:
    for box in r.boxes:
        print(f"Confidence: {box.conf.item():.2f} | BBox: {box.xyxy[0].tolist()}")