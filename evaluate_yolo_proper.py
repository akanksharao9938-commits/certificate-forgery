from ultralytics import YOLO
import os

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = r"D:\certificate-forgery-backend(2)\runs\certificate_forgery\weights\best.pt"

DATASET_YAML = r"D:\kaggle-datasets\certificate-forgery\d1Certificate forgery detection\data.yaml"

# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"YOLO model not found:\n{MODEL_PATH}"
    )

if not os.path.exists(DATASET_YAML):
    raise FileNotFoundError(
        f"Dataset YAML not found:\n{DATASET_YAML}"
    )

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("YOLO11 PROPER TEST EVALUATION")
print("=" * 70)

print("\nLoading YOLO11 model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")

# ============================================================
# VALIDATE ON ORIGINAL TEST SET
# ============================================================

print("\nRunning evaluation on original YOLO test dataset...")
print("This may take some time on CPU.\n")

metrics = model.val(
    data=DATASET_YAML,
    split="test",
    imgsz=416,
    batch=1,
    device="cpu",
    workers=0,
    plots=True,
    verbose=True
)

# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("YOLO11 PROPER TEST RESULTS")
print("=" * 70)

print(f"\nPrecision : {metrics.box.mp:.4f}")
print(f"Recall    : {metrics.box.mr:.4f}")
print(f"mAP@50    : {metrics.box.map50:.4f}")
print(f"mAP@50-95 : {metrics.box.map:.4f}")

print("\nAs percentages:")

print(f"Precision : {metrics.box.mp * 100:.2f}%")
print(f"Recall    : {metrics.box.mr * 100:.2f}%")
print(f"mAP@50    : {metrics.box.map50 * 100:.2f}%")
print(f"mAP@50-95 : {metrics.box.map * 100:.2f}%")

# ============================================================
# PER-CLASS RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("PER-CLASS RESULTS")
print("=" * 70)

print("\nClass names:")

for class_id, class_name in model.names.items():
    print(f"{class_id} = {class_name}")

print("\nPer-class mAP@50:")

try:
    for class_id, class_name in model.names.items():
        print(
            f"{class_name}: "
            f"{metrics.box.ap50[class_id] * 100:.2f}%"
        )
except Exception as e:
    print("Per-class metrics could not be displayed:", e)

# ============================================================
# SAVE LOCATION
# ============================================================

print("\n")
print("=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)

print("\nYOLO training/evaluation results are available in:")
print(r"D:\certificate-forgery-backend(2)\runs\certificate_forgery")