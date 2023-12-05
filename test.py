from ultralytics import YOLO



if __name__ == "__main__":
    model = YOLO("./runs/detect/train/weights/best.pt")
    results = model.predict(source="./dhc-2_cockpit.png", save=True)