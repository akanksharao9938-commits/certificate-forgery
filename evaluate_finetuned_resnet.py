import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model

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

BASE_DIR = r"D:\certificate-forgery-backend(2)"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "resnet_certificate_model_finetuned.keras"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "resnet_dataset",
    "test"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("RESNET50 TEST EVALUATION")
print("=" * 70)

print("\nLoading model...")

model = load_model(
    MODEL_PATH
)

print("Model loaded successfully.")

print("\nModel:")
print(MODEL_PATH)


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEST DATASET")
print("=" * 70)

test_dataset = tf.keras.utils.image_dataset_from_directory(

    TEST_DIR,

    image_size=IMAGE_SIZE,

    batch_size=1,

    label_mode="binary",

    shuffle=False
)

print("\nClass names:")
print(test_dataset.class_names)


# ============================================================
# PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("RUNNING PREDICTIONS")
print("=" * 70)

y_true = []
y_pred = []
probabilities = []


for images, labels in test_dataset:

    prediction = model.predict(
        images,
        verbose=0
    )

    # Model output is a single sigmoid probability.
    probability = float(
        np.asarray(prediction).reshape(-1)[0]
    )

    # 0 = authentic
    # 1 = forged
    predicted_class = (
        1
        if probability >= 0.5
        else 0
    )

    # Convert TensorFlow label safely to a scalar.
    true_class = int(
        np.asarray(labels.numpy()).reshape(-1)[0]
    )

    y_true.append(
        true_class
    )

    y_pred.append(
        predicted_class
    )

    probabilities.append(
        probability
    )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    f"\nTest images: {len(y_true)}"
)

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1-score : {f1 * 100:.2f}%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("\nRows = Actual")
print("Columns = Predicted")

print(
    "\n              Authentic  Forged"
)

print(
    f"Authentic     {cm[0][0]:8d}  {cm[0][1]:6d}"
)

print(
    f"Forged        {cm[1][0]:8d}  {cm[1][1]:6d}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "Authentic",
            "Forged"
        ],
        zero_division=0
    )
)


# ============================================================
# PREDICTION COUNTS
# ============================================================

authentic_predictions = sum(
    1
    for p in y_pred
    if p == 0
)

forged_predictions = sum(
    1
    for p in y_pred
    if p == 1
)

actual_authentic = sum(
    1
    for p in y_true
    if p == 0
)

actual_forged = sum(
    1
    for p in y_true
    if p == 1
)


print("\n" + "=" * 70)
print("DATASET / PREDICTION COUNTS")
print("=" * 70)

print(
    f"\nActual Authentic: "
    f"{actual_authentic}"
)

print(
    f"Actual Forged: "
    f"{actual_forged}"
)

print(
    f"\nPredicted Authentic: "
    f"{authentic_predictions}"
)

print(
    f"Predicted Forged: "
    f"{forged_predictions}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)