from ultralytics import YOLO
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = r"D:\certificate-forgery-backend(2)\runs\certificate_forgery\weights\best.pt"

TEST_PATH = r"D:\certificate-forgery-backend(2)\resnet_dataset\test"

# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"YOLO model not found:\n{MODEL_PATH}"
    )

print("=" * 70)
print("YOLO11 TEST EVALUATION")
print("=" * 70)

print("\nLoading YOLO11 model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")

# ============================================================
# GET TEST IMAGES
# ============================================================

image_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

test_images = []

for class_name in ["authentic", "forged"]:

    class_folder = os.path.join(
        TEST_PATH,
        class_name
    )

    if not os.path.exists(class_folder):
        continue

    for filename in os.listdir(class_folder):

        if filename.lower().endswith(image_extensions):

            image_path = os.path.join(
                class_folder,
                filename
            )

            test_images.append(
                (image_path, class_name)
            )

print("\nTest images found:", len(test_images))

# ============================================================
# PREDICTIONS
# ============================================================

y_true = []
y_pred = []

print("\nRunning YOLO predictions...")

for index, (image_path, true_class) in enumerate(test_images, start=1):

    results = model.predict(
        source=image_path,
        imgsz=416,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    # --------------------------------------------------------
    # Determine YOLO prediction
    # --------------------------------------------------------

    predicted_forged = False

    if result.boxes is not None and len(result.boxes) > 0:

        for box in result.boxes:

            class_id = int(box.cls[0])

            class_name = model.names[class_id].lower()

            if class_name == "fake":
                predicted_forged = True
                break

    if predicted_forged:
        prediction = "forged"
    else:
        prediction = "authentic"

    y_true.append(true_class)
    y_pred.append(prediction)

    print(
        f"[{index:02d}/{len(test_images)}] "
        f"Actual: {true_class:9s} | "
        f"Predicted: {prediction}"
    )

# ============================================================
# CONVERT LABELS
# ============================================================

label_map = {
    "authentic": 0,
    "forged": 1
}

y_true_numeric = np.array([
    label_map[x]
    for x in y_true
])

y_pred_numeric = np.array([
    label_map[x]
    for x in y_pred
])

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true_numeric,
    y_pred_numeric
)

precision = precision_score(
    y_true_numeric,
    y_pred_numeric,
    zero_division=0
)

recall = recall_score(
    y_true_numeric,
    y_pred_numeric,
    zero_division=0
)

f1 = f1_score(
    y_true_numeric,
    y_pred_numeric,
    zero_division=0
)

cm = confusion_matrix(
    y_true_numeric,
    y_pred_numeric
)

# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("YOLO11 TEST RESULTS")
print("=" * 70)

print(f"\nTest images : {len(test_images)}")
print(f"Accuracy    : {accuracy * 100:.2f}%")
print(f"Precision   : {precision * 100:.2f}%")
print(f"Recall      : {recall * 100:.2f}%")
print(f"F1-Score    : {f1 * 100:.2f}%")

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    """
                 Predicted
              Authentic  Forged
Actual Authentic
Actual Forged
"""
)

print(cm)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_true_numeric,
        y_pred_numeric,
        target_names=[
            "authentic",
            "forged"
        ],
        zero_division=0
    )
)

# ============================================================
# SUMMARY
# ============================================================

authentic_count = y_pred.count("authentic")
forged_count = y_pred.count("forged")

print("\n")
print("=" * 70)
print("PREDICTION SUMMARY")
print("=" * 70)

print(f"\nPredicted Authentic : {authentic_count}")
print(f"Predicted Forged    : {forged_count}")

print("\n")
print("=" * 70)
print("YOLO11 EVALUATION COMPLETED")
print("=" * 70)