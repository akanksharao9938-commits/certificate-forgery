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


# ==================================================
# PATHS
# ==================================================

BASE_DIR = r"D:\certificate-forgery-backend(2)"

TEST_DIR = os.path.join(
    BASE_DIR,
    "resnet_dataset",
    "test"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "resnet_certificate_model_finetuned.keras"
)


# ==================================================
# SETTINGS
# ==================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 4


# ==================================================
# CHECK MODEL
# ==================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Fine-tuned model not found:\n{MODEL_PATH}"
    )


# ==================================================
# LOAD MODEL
# ==================================================

print()
print("Loading fine-tuned ResNet50 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Fine-tuned model loaded successfully.")


# ==================================================
# LOAD TEST DATA
# ==================================================

print()
print("Loading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(

    TEST_DIR,

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="binary",

    shuffle=False
)


class_names = test_dataset.class_names

print()
print("Class names:")
print(class_names)


# ==================================================
# PREDICTIONS
# ==================================================

y_true = []

y_probability = []


for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    y_probability.extend(
        predictions.flatten()
    )

    y_true.extend(
        labels.numpy().flatten()
    )


y_true = np.array(
    y_true
).astype(int)


y_probability = np.array(
    y_probability
)


# ==================================================
# CLASSIFICATION
# ==================================================

y_pred = (
    y_probability >= 0.5
).astype(int)


# ==================================================
# METRICS
# ==================================================

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


# ==================================================
# CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


# ==================================================
# RESULTS
# ==================================================

print()
print("==========================================")
print("FINE-TUNED RESNET50 TEST RESULTS")
print("==========================================")

print()

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1-Score  : {f1 * 100:.2f}%"
)


# ==================================================
# CONFUSION MATRIX
# ==================================================

print()
print("==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print(cm)


# ==================================================
# CLASSIFICATION REPORT
# ==================================================

print()
print("==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)


# ==================================================
# CLASS MAPPING
# ==================================================

print()
print("Class mapping:")

for index, class_name in enumerate(class_names):

    print(
        f"{index} = {class_name}"
    )


print()
print("Fine-tuned model evaluation completed!")