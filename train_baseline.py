from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # nano model for stability

    model.train(
        data="data/data.yaml",
        epochs=20,
        imgsz=640,
        batch=4,
        workers=1,
        cache=False,
        name="baseline_yolo",
        project="models"
    )

if __name__ == "__main__":
    main()
