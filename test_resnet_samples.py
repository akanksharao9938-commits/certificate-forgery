import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\certificate-forgery-backend(2)"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "resnet_certificate_model_finetuned.keras"
)

AUTHENTIC_DIR = os.path.join(
    BASE_DIR,
    "resnet_dataset",
    "train",
    "authentic"
)

FORGED_DIR = os.path.join(
    BASE_DIR,
    "resnet_dataset",
    "train",
    "forged"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("LOADING RESNET50 MODEL")
print("=" * 70)

model = load_model(MODEL_PATH)

print("\nModel loaded successfully:")
print(MODEL_PATH)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image_path):

    image = tf.keras.utils.load_img(
        image_path,
        target_size=(224, 224)
    )

    image_array = tf.keras.utils.img_to_array(
        image
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    probability = float(
        model.predict(
            image_array,
            verbose=0
        )[0][0]
    )

    if probability >= 0.5:

        prediction = "Forged"
        confidence = probability * 100

    else:

        prediction = "Authentic"
        confidence = (1.0 - probability) * 100

    return prediction, confidence, probability


# ============================================================
# AUTHENTIC SAMPLES
# ============================================================

print("\n" + "=" * 70)
print("AUTHENTIC TRAINING SAMPLES")
print("=" * 70)

authentic_files = [
    f
    for f in os.listdir(AUTHENTIC_DIR)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
][:10]


for filename in authentic_files:

    image_path = os.path.join(
        AUTHENTIC_DIR,
        filename
    )

    prediction, confidence, probability = predict_image(
        image_path
    )

    print(
        f"{filename} -> "
        f"{prediction} "
        f"(confidence: {confidence:.2f}%, "
        f"raw probability: {probability:.4f})"
    )


# ============================================================
# FORGED SAMPLES
# ============================================================

print("\n" + "=" * 70)
print("FORGED TRAINING SAMPLES")
print("=" * 70)

forged_files = [
    f
    for f in os.listdir(FORGED_DIR)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
][:10]


for filename in forged_files:

    image_path = os.path.join(
        FORGED_DIR,
        filename
    )

    prediction, confidence, probability = predict_image(
        image_path
    )

    print(
        f"{filename} -> "
        f"{prediction} "
        f"(confidence: {confidence:.2f}%, "
        f"raw probability: {probability:.4f})"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("TEST COMPLETED")
print("=" * 70)