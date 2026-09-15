import os
import numpy as np
import tensorflow as tf

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

DATASET_PATH = r"D:\certificate-forgery-backend(2)\resnet_dataset"

MODEL_PATH = r"D:\certificate-forgery-backend(2)\model\resnet_certificate_model_all_data.keras"

TEST_PATH = os.path.join(
    DATASET_PATH,
    "test"
)

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 4


# ============================================================
# START
# ============================================================

print("=" * 70)
print("RESNET50 TEST EVALUATION")
print("=" * 70)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading ResNet50 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(

    TEST_PATH,

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="binary",

    shuffle=False
)


print("\nClass names:")

print(
    test_dataset.class_names
)


# ============================================================
# GET TRUE LABELS
# ============================================================

y_true = []

y_probability = []


print("\nRunning predictions...")


for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    y_true.extend(
        labels.numpy().flatten().astype(int)
    )

    y_probability.extend(
        predictions.flatten()
    )


# ============================================================
# CONVERT PROBABILITIES TO CLASSES
# ============================================================

y_true = np.array(
    y_true
)

y_probability = np.array(
    y_probability
)


y_pred = (
    y_probability >= 0.5
).astype(int)


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
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)


print(
    f"\nTest images : {len(y_true)}"
)

print(
    f"Accuracy    : {accuracy * 100:.2f}%"
)

print(
    f"Precision   : {precision * 100:.2f}%"
)

print(
    f"Recall      : {recall * 100:.2f}%"
)

print(
    f"F1-Score    : {f1 * 100:.2f}%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    "\n                 Predicted"
)

print(
    "              Authentic  Forged"
)

print(
    f"Actual Authentic    {cm[0][0]:3d}       {cm[0][1]:3d}"
)

print(
    f"Actual Forged       {cm[1][0]:3d}       {cm[1][1]:3d}"
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
            "authentic",
            "forged"
        ],
        zero_division=0
    )
)


# ============================================================
# INDIVIDUAL PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION SUMMARY")
print("=" * 70)


authentic_predictions = np.sum(
    y_pred == 0
)

forged_predictions = np.sum(
    y_pred == 1
)


print(
    f"\nPredicted Authentic : "
    f"{authentic_predictions}"
)

print(
    f"Predicted Forged    : "
    f"{forged_predictions}"
)


print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)