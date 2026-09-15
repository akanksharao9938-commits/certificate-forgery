from ultralytics import YOLO
import os

# Dataset
DATASET_PATH = r"D:\kaggle-datasets\certificate-forgery\d1Certificate forgery detection\data.yaml"

# Small pretrained YOLO model
MODEL_PATH = "yolo11n.pt"

# Keep training output on D:
PROJECT_PATH = r"D:\certificate-forgery-backend(2)\runs"

# Load model
model = YOLO(MODEL_PATH)

# Train
results = model.train(
    data=DATASET_PATH,

    # Fewer epochs for initial training
    epochs=15,

    # Smaller image size = much lower RAM usage
    imgsz=416,

    # IMPORTANT: one image at a time
    batch=1,

    # CPU
    device="cpu",

    # Don't use multiple worker processes
    workers=0,

    # Reduce augmentation/memory usage
    cache=False,

    # Save results on D:
    project=PROJECT_PATH,
    name="certificate_forgery",
    exist_ok=True,

    # Save best model
    save=True,

    # Validate during training
    val=True,

    # Stop if validation stops improving
    patience=5
)

print("\nTraining completed successfully!")

print(
    "\nBest model should be located at:"
)

print(
    r"D:\certificate-forgery-backend(2)\runs\certificate_forgery\weights\best.pt"
)