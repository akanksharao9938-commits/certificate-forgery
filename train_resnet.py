import os
import tensorflow as tf
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = r"D:\certificate-forgery-backend(2)\resnet_dataset"

MODEL_FOLDER = r"D:\certificate-forgery-backend(2)\model"

MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "resnet_certificate_model_all_data.keras"
)

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 4

EPOCHS = 20

SEED = 42


# ============================================================
# CREATE MODEL FOLDER
# ============================================================

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


# ============================================================
# GPU / CPU INFORMATION
# ============================================================

print("=" * 70)
print("RESNET50 CERTIFICATE FORGERY DETECTION")
print("=" * 70)

print("\nTensorFlow version:")
print(tf.__version__)

gpus = tf.config.list_physical_devices("GPU")

if gpus:

    print("\nGPU detected:")
    print(gpus)

else:

    print("\nNo GPU detected.")
    print("Training will use CPU.")


# ============================================================
# LOAD TRAIN DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING TRAINING DATA")
print("=" * 70)

train_dataset = tf.keras.utils.image_dataset_from_directory(

    os.path.join(
        DATASET_PATH,
        "train"
    ),

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="binary",

    shuffle=True,

    seed=SEED
)


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING VALIDATION DATA")
print("=" * 70)

valid_dataset = tf.keras.utils.image_dataset_from_directory(

    os.path.join(
        DATASET_PATH,
        "valid"
    ),

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="binary",

    shuffle=False
)


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEST DATA")
print("=" * 70)

test_dataset = tf.keras.utils.image_dataset_from_directory(

    os.path.join(
        DATASET_PATH,
        "test"
    ),

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="binary",

    shuffle=False
)


# ============================================================
# CLASS NAMES
# ============================================================

print("\n" + "=" * 70)
print("CLASS INFORMATION")
print("=" * 70)

print("\nClass names:")

print(
    train_dataset.class_names
)

print(
    "\nIMPORTANT:"
)

print(
    "0 = authentic"
)

print(
    "1 = forged"
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.03
    ),

    layers.RandomZoom(
        0.10
    ),

    layers.RandomContrast(
        0.10
    )

])


# ============================================================
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    AUTOTUNE
)

valid_dataset = valid_dataset.prefetch(
    AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    AUTOTUNE
)


# ============================================================
# CALCULATE CLASS WEIGHTS
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING CLASS WEIGHTS")
print("=" * 70)


authentic_count = 78

forged_count = 99

total_count = (
    authentic_count
    + forged_count
)

class_weight_0 = (
    total_count
    / (
        2 * authentic_count
    )
)

class_weight_1 = (
    total_count
    / (
        2 * forged_count
    )
)

class_weights = {

    0: class_weight_0,

    1: class_weight_1

}


print(
    f"\nAuthentic images: {authentic_count}"
)

print(
    f"Forged images: {forged_count}"
)

print(
    f"\nClass weight - Authentic (0): "
    f"{class_weight_0:.4f}"
)

print(
    f"Class weight - Forged (1): "
    f"{class_weight_1:.4f}"
)


# ============================================================
# LOAD PRETRAINED RESNET50
# ============================================================

print("\n" + "=" * 70)
print("LOADING RESNET50")
print("=" * 70)

base_model = ResNet50(

    weights="imagenet",

    include_top=False,

    input_shape=(
        224,
        224,
        3
    )
)


# ============================================================
# FREEZE RESNET50
# ============================================================

base_model.trainable = False


# ============================================================
# BUILD MODEL
# ============================================================

inputs = tf.keras.Input(
    shape=(
        224,
        224,
        3
    )
)


x = data_augmentation(
    inputs
)


# ResNet50 preprocessing
x = tf.keras.applications.resnet50.preprocess_input(
    x
)


x = base_model(
    x,
    training=False
)


x = layers.GlobalAveragePooling2D()(
    x
)


x = layers.Dropout(
    0.30
)(
    x
)


outputs = layers.Dense(
    1,
    activation="sigmoid"
)(
    x
)


model = tf.keras.Model(
    inputs,
    outputs
)


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="binary_crossentropy",

    metrics=[

        "accuracy",

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        )

    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL SUMMARY")
print("=" * 70)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_loss",

    save_best_only=True,

    verbose=1
)


early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True,

    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print(
    f"\nEpochs: {EPOCHS}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    "Image size: 224 x 224"
)

print(
    "\nTraining ResNet50..."
)


history = model.fit(

    train_dataset,

    validation_data=valid_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=[

        checkpoint,

        early_stopping

    ]

)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    MODEL_PATH
)


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print("\nModel saved at:")

print(
    MODEL_PATH
)

print("\nThe next step is evaluation on the")
print("40-image test dataset.")