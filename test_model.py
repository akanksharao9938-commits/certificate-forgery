from ultralytics import YOLO

MODEL_PATH = r"D:\certificate-forgery-backend(2)\runs\certificate_forgery\weights\best.pt"

TEST_IMAGES = r"D:\kaggle-datasets\certificate-forgery\d1Certificate forgery detection\test\images"

model = YOLO(MODEL_PATH)

results = model.predict(
    source=TEST_IMAGES,
    imgsz=416,
    conf=0.25,
    save=True,
    save_txt=True,
    project=r"D:\certificate-forgery-backend(2)\test_results",
    name="predictions",
    exist_ok=True
)

print("\nTesting completed!")
print("Results saved to:")
print(r"D:\certificate-forgery-backend(2)\test_results\predictions")