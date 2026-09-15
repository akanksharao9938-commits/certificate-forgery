import os
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


# ==================================================
# PATHS
# ==================================================

BASE_DIR = r"D:\certificate-forgery-backend(2)"

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "resnet_dataset",
    "train"
)

VALID_DIR = os.path.join(
    BASE_DIR,
    "resnet_dataset",
    "valid"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "resnet_certificate_model.keras"
)

FINE_TUNED_MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "resnet_certificate_model_finetuned.keras"
)


# ==================================================
# SETTINGS
# ==================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 4

EPOCHS = 10

SEED = 42


# ==================================================
# LOAD DATA
# ==================================================

print()
print("Loading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    seed=SEED,
    shuffle=True
)

print()
print("Loading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    seed=SEED,
    shuffle=False
)


print()
print("Class names:")
print(train_dataset.class_names)


# ==================================================
# CLASS WEIGHTS
# ==================================================

class_names = train_dataset.class_names

class_counts = []

for class_name in class_names:

    folder = os.path.join(
        TRAIN_DIR,
        class_name
    )

    count = len([
        f for f in os.listdir(folder)
        if os.path.isfile(
            os.path.join(folder, f)
        )
    ])

    class_counts.append(count)


print()
print("Training class counts:")

for name, count in zip(
    class_names,
    class_counts
):

    print(
        name,
        ":",
        count
    )


classes = np.array([0, 1])

class_weights_values = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=np.concatenate([
        np.full(class_counts[0], 0),
        np.full(class_counts[1], 1)
    ])
)

class_weights = {
    0: float(class_weights_values[0]),
    1: float(class_weights_values[1])
}


print()
print("Class weights:")
print(class_weights)


# ==================================================
# PERFORMANCE
# ==================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ==================================================
# LOAD PREVIOUS RESNET MODEL
# ==================================================

print()
print("Loading previous ResNet50 model...")

model = load_model(
    MODEL_PATH
)

print("Previous model loaded successfully.")


# ==================================================
# FIND RESNET50 BACKBONE
# ==================================================

resnet_base = None

for layer in model.layers:

    if layer.name == "resnet50":

        resnet_base = layer

        break


if resnet_base is None:

    raise ValueError(
        "ResNet50 backbone was not found in the model."
    )


# ==================================================
# UNFREEZE UPPER RESNET LAYERS
# ==================================================

resnet_base.trainable = True


# Freeze most layers

for layer in resnet_base.layers:

    layer.trainable = False


# Unfreeze last 30 layers

for layer in resnet_base.layers[-30:]:

    layer.trainable = True


print()
print("ResNet50 fine-tuning configuration:")
print("Most ResNet layers: Frozen")
print("Last 30 ResNet layers: Trainable")


# ==================================================
# COMPILE WITH SMALL LEARNING RATE
# ==================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy"
    ]

)


# ==================================================
# CALLBACKS
# ==================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=4,

        restore_best_weights=True

    ),

    tf.keras.callbacks.ModelCheckpoint(

        FINE_TUNED_MODEL_PATH,

        monitor="val_accuracy",

        save_best_only=True

    )

]


# ==================================================
# TRAIN
# ==================================================

print()
print("==========================================")
print("Starting ResNet50 fine-tuning...")
print("==========================================")
print()


history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks

)


# ==================================================
# SAVE FINAL MODEL
# ==================================================

model.save(
    FINE_TUNED_MODEL_PATH
)


print()
print("==========================================")
print("Fine-tuning completed successfully!")
print("==========================================")

print()
print("Fine-tuned model saved at:")

print(
    FINE_TUNED_MODEL_PATH
)