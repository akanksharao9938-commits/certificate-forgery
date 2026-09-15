import os
import shutil
import random

# ============================================================
# PATHS
# ============================================================

SOURCE_DATASET = r"D:\kaggle-datasets\certificate-forgery\d1Certificate forgery detection"

OUTPUT_DATASET = r"D:\certificate-forgery-backend(2)\resnet_dataset"

SOURCE_SPLITS = ["train", "valid", "test"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_image_files(folder):

    if not os.path.exists(folder):
        return []

    images = []

    for filename in os.listdir(folder):

        file_path = os.path.join(
            folder,
            filename
        )

        if os.path.isfile(file_path):

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension in IMAGE_EXTENSIONS:
                images.append(file_path)

    return images


def get_class_from_label(label_path):

    """
    YOLO dataset mapping:

        0 = fake
        1 = true

    If any class 0 exists:
        forged

    If only class 1 exists:
        authentic
    """

    if not os.path.exists(label_path):
        return None

    try:

        with open(
            label_path,
            "r"
        ) as file:

            lines = file.readlines()

        classes = []

        for line in lines:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                continue

            try:

                class_id = int(parts[0])

                if class_id in [0, 1]:

                    classes.append(class_id)

            except ValueError:

                continue

        if not classes:
            return None

        # Any fake region = forged certificate
        if 0 in classes:
            return "forged"

        # Only true regions = authentic certificate
        if all(class_id == 1 for class_id in classes):
            return "authentic"

        return None

    except Exception as error:

        print(
            f"Error reading label: {label_path}"
        )

        print(error)

        return None


# ============================================================
# START
# ============================================================

print("=" * 70)
print("RESNET DATASET PREPARATION")
print("=" * 70)

print("\nSource:")
print(SOURCE_DATASET)

print("\nOutput:")
print(OUTPUT_DATASET)


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(SOURCE_DATASET):

    raise FileNotFoundError(
        f"\nDataset not found:\n{SOURCE_DATASET}"
    )


# ============================================================
# COLLECT IMAGES
# ============================================================

forged_images = []

authentic_images = []

raw_image_count = 0

missing_label_count = 0

invalid_label_count = 0


print("\n" + "=" * 70)
print("SCANNING DATASET")
print("=" * 70)


for split in SOURCE_SPLITS:

    images_folder = os.path.join(
        SOURCE_DATASET,
        split,
        "images"
    )

    labels_folder = os.path.join(
        SOURCE_DATASET,
        split,
        "labels"
    )

    print(f"\nScanning: {split}")

    if not os.path.exists(images_folder):

        print(
            f"Images folder not found: {images_folder}"
        )

        continue

    images = get_image_files(
        images_folder
    )

    print(
        f"Images found: {len(images)}"
    )

    for image_path in images:

        raw_image_count += 1

        filename = os.path.basename(
            image_path
        )

        image_name = os.path.splitext(
            filename
        )[0]

        label_path = os.path.join(
            labels_folder,
            image_name + ".txt"
        )

        # ----------------------------------------------------
        # LABEL CHECK
        # ----------------------------------------------------

        if not os.path.exists(label_path):

            missing_label_count += 1

            continue

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        class_name = get_class_from_label(
            label_path
        )

        if class_name is None:

            invalid_label_count += 1

            continue

        # ----------------------------------------------------
        # ADD IMAGE
        # ----------------------------------------------------

        if class_name == "forged":

            forged_images.append(
                image_path
            )

        elif class_name == "authentic":

            authentic_images.append(
                image_path
            )


# ============================================================
# DATASET SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(
    f"\nRaw image files found       : {raw_image_count}"
)

print(
    f"Missing label files         : {missing_label_count}"
)

print(
    f"Invalid/empty labels        : {invalid_label_count}"
)

print(
    f"\nForged images               : {len(forged_images)}"
)

print(
    f"Authentic images            : {len(authentic_images)}"
)

total_valid = (
    len(forged_images)
    + len(authentic_images)
)

print(
    f"Total valid images          : {total_valid}"
)


# ============================================================
# CHECK
# ============================================================

if len(forged_images) == 0:

    raise ValueError(
        "No forged images were found."
    )

if len(authentic_images) == 0:

    raise ValueError(
        "No authentic images were found."
    )


print("\nUsing ALL valid labeled images.")

print(
    "No undersampling or duplicate removal will be performed."
)


# ============================================================
# SHUFFLE
# ============================================================

random.shuffle(
    forged_images
)

random.shuffle(
    authentic_images
)


# ============================================================
# STRATIFIED 70 / 15 / 15 SPLIT
# ============================================================

def split_class(images):

    total = len(images)

    train_count = int(
        total * 0.70
    )

    valid_count = int(
        total * 0.15
    )

    train = images[
        :train_count
    ]

    valid = images[
        train_count:
        train_count + valid_count
    ]

    test = images[
        train_count + valid_count:
    ]

    return train, valid, test


(
    forged_train,
    forged_valid,
    forged_test
) = split_class(
    forged_images
)


(
    authentic_train,
    authentic_valid,
    authentic_test
) = split_class(
    authentic_images
)


# ============================================================
# COMBINE
# ============================================================

train_images = []

for image in forged_train:

    train_images.append(
        ("forged", image)
    )

for image in authentic_train:

    train_images.append(
        ("authentic", image)
    )


valid_images = []

for image in forged_valid:

    valid_images.append(
        ("forged", image)
    )

for image in authentic_valid:

    valid_images.append(
        ("authentic", image)
    )


test_images = []

for image in forged_test:

    test_images.append(
        ("forged", image)
    )

for image in authentic_test:

    test_images.append(
        ("authentic", image)
    )


random.shuffle(
    train_images
)

random.shuffle(
    valid_images
)

random.shuffle(
    test_images
)


# ============================================================
# REMOVE OLD OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("PREPARING OUTPUT FOLDER")
print("=" * 70)


if os.path.exists(
    OUTPUT_DATASET
):

    print(
        "\nRemoving old ResNet dataset..."
    )

    shutil.rmtree(
        OUTPUT_DATASET
    )


# ============================================================
# CREATE FOLDERS
# ============================================================

for split in [
    "train",
    "valid",
    "test"
]:

    for class_name in [
        "authentic",
        "forged"
    ]:

        folder = os.path.join(
            OUTPUT_DATASET,
            split,
            class_name
        )

        os.makedirs(
            folder,
            exist_ok=True
        )


# ============================================================
# COPY FUNCTION
# ============================================================

def copy_images(
    image_list,
    split_name
):

    for index, (
        class_name,
        source_path
    ) in enumerate(
        image_list,
        start=1
    ):

        original_filename = os.path.basename(
            source_path
        )

        extension = os.path.splitext(
            original_filename
        )[1].lower()

        destination_folder = os.path.join(
            OUTPUT_DATASET,
            split_name,
            class_name
        )

        # Unique name based on split + class + index
        new_filename = (
            f"{split_name}_"
            f"{class_name}_"
            f"{index:05d}"
            f"{extension}"
        )

        destination_path = os.path.join(
            destination_folder,
            new_filename
        )

        shutil.copy2(
            source_path,
            destination_path
        )


# ============================================================
# COPY DATA
# ============================================================

print("\n" + "=" * 70)
print("COPYING DATA")
print("=" * 70)


print("\nCopying training images...")

copy_images(
    train_images,
    "train"
)


print("Copying validation images...")

copy_images(
    valid_images,
    "valid"
)


print("Copying test images...")

copy_images(
    test_images,
    "test"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESNET DATASET")
print("=" * 70)


print("\nTRAIN")

print(
    f"Forged    : {len(forged_train)}"
)

print(
    f"Authentic : {len(authentic_train)}"
)

print(
    f"Total     : {len(train_images)}"
)


print("\nVALIDATION")

print(
    f"Forged    : {len(forged_valid)}"
)

print(
    f"Authentic : {len(authentic_valid)}"
)

print(
    f"Total     : {len(valid_images)}"
)


print("\nTEST")

print(
    f"Forged    : {len(forged_test)}"
)

print(
    f"Authentic : {len(authentic_test)}"
)

print(
    f"Total     : {len(test_images)}"
)


print("\n" + "=" * 70)
print("DATASET CREATION COMPLETED")
print("=" * 70)

print("\nSaved at:")

print(
    OUTPUT_DATASET
)

print("\nDataset structure:")

print(
r"""
resnet_dataset/
│
├── train/
│   ├── authentic/
│   └── forged/
│
├── valid/
│   ├── authentic/
│   └── forged/
│
└── test/
    ├── authentic/
    └── forged/
"""
)

print("\nAll valid labeled images were used.")

print(
    "Split ratio: 70% train / 15% validation / 15% test"
)

print(
    "Random seed: 42"
)

print("\nNext step:")

print(
    "Train the ResNet50 model on the new dataset."
)