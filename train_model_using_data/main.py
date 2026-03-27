from ultralytics import YOLO

model = YOLO("weights/yolo11n.pt")



def train():
    results = model.train(
        data="face_dataset/face.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device="cpu",       
        patience=20,       
        name="face_detector"
    )

if __name__ == "__main__":

    

    train()