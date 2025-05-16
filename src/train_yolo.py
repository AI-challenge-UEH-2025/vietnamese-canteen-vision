from ultralytics import YOLO
import torch

def train_yolo():
    # Check GPU
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")

    # Load model
    model = YOLO('yolov8-seg.pt')

    # Train
    model.train(
        data='data/an com.v11-train-lan-3.yolov8/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        patience=20,
        augment=True,
        val=True,
        save=True,
        project='runs/train',
        name='yolo_food',
        optimizer='Adam',
        lr0=0.001,
        weight_decay=1e-4,
        cos_lr=True,
        mosaic=0.8,
        flipud=0.5,
        fliplr=0.5,
        degrees=15,
        translate=0.1,
        scale=0.5,
        shear=0.1
    )
    model.save('models/best.pt')

if __name__ == "__main__":
    train_yolo()