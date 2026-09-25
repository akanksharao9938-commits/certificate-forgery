import os
import uuid
import shutil

import cv2
import numpy as np
import tensorflow as tf

from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\certificate-forgery-backend(2)"

YOLO_MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs",
    "certificate_forgery",
    "weights",
    "best.pt"
)

RESNET_MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "resnet_certificate_model_finetuned.keras"
)

RESULTS_FOLDER = os.path.join(
    BASE_DIR,
    "results",
    "predictions"
)

os.makedirs(
    RESULTS_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD YOLO11
# ============================================================

print("=" * 70)
print("LOADING YOLO11 MODEL")
print("=" * 70)

if not os.path.exists(YOLO_MODEL_PATH):

    raise FileNotFoundError(
        f"YOLO11 model not found:\n{YOLO_MODEL_PATH}"
    )

yolo_model = YOLO(
    YOLO_MODEL_PATH
)

print(
    f"YOLO11 loaded:\n{YOLO_MODEL_PATH}"
)


# ============================================================
# LOAD RESNET50
# ============================================================

print("\n" + "=" * 70)
print("LOADING RESNET50 MODEL")
print("=" * 70)

if not os.path.exists(RESNET_MODEL_PATH):

    raise FileNotFoundError(
        f"ResNet50 model not found:\n{RESNET_MODEL_PATH}"
    )

resnet_model = tf.keras.models.load_model(
    RESNET_MODEL_PATH
)

print(
    f"ResNet50 loaded:\n{RESNET_MODEL_PATH}"
)


# ============================================================
# RESNET50 PREDICTION
# ============================================================

def predict_with_resnet(image_path):

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

    prediction = resnet_model.predict(
        image_array,
        verbose=0
    )

    forged_probability = float(
        np.asarray(prediction).reshape(-1)[0]
    )

    authentic_probability = (
        1.0 - forged_probability
    )

    if forged_probability >= 0.5:

        resnet_prediction = "Forged"

        resnet_confidence = (
            forged_probability * 100
        )

    else:

        resnet_prediction = "Authentic"

        resnet_confidence = (
            authentic_probability * 100
        )

    return {
        "prediction": resnet_prediction,
        "confidence": round(
            resnet_confidence,
            2
        ),
        "forged_probability": round(
            forged_probability,
            4
        ),
        "authentic_probability": round(
            authentic_probability,
            4
        )
    }


# ============================================================
# YOLO11 PREDICTION
# ============================================================

def predict_with_yolo(image_path):

    prediction_results = yolo_model.predict(

        source=image_path,

        imgsz=416,

        conf=0.25,

        device="cpu",

        verbose=False,

        save=True,

        project=RESULTS_FOLDER,

        name="yolo_output",

        exist_ok=True
    )

    result = prediction_results[0]

    detections = []

    fake_confidences = []

    true_confidences = []

    # --------------------------------------------------------
    # PROCESS DETECTIONS
    # --------------------------------------------------------

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(
                box.cls[0].item()
            )

            confidence = float(
                box.conf[0].item()
            )

            coordinates = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .tolist()
            )

            if class_id == 0:

                class_name = "fake"

                fake_confidences.append(
                    confidence
                )

            elif class_id == 1:

                class_name = "true"

                true_confidences.append(
                    confidence
                )

            else:

                class_name = str(
                    class_id
                )

            detections.append({

                "class": class_name,

                "confidence": round(
                    confidence * 100,
                    2
                ),

                "bounding_box": {
                    "x1": round(
                        coordinates[0],
                        2
                    ),
                    "y1": round(
                        coordinates[1],
                        2
                    ),
                    "x2": round(
                        coordinates[2],
                        2
                    ),
                    "y2": round(
                        coordinates[3],
                        2
                    )
                }
            })


    # --------------------------------------------------------
    # YOLO SUMMARY
    # --------------------------------------------------------

    fake_region_count = len(
        fake_confidences
    )

    true_region_count = len(
        true_confidences
    )

    if fake_region_count > 0:

        yolo_prediction = (
            "Suspicious region detected"
        )

        yolo_confidence = max(
            fake_confidences
        ) * 100

    elif true_region_count > 0:

        yolo_prediction = (
            "Certificate regions detected as true"
        )

        yolo_confidence = max(
            true_confidences
        ) * 100

    else:

        yolo_prediction = (
            "No certificate regions detected"
        )

        yolo_confidence = 0.0


    # --------------------------------------------------------
    # RESULT IMAGE
    # --------------------------------------------------------

    result_image_url = None

    saved_image_path = getattr(
        result,
        "save_dir",
        None
    )

    if saved_image_path:

        possible_image = os.path.join(
            saved_image_path,
            os.path.basename(image_path)
        )

        if os.path.exists(
            possible_image
        ):

            unique_name = (
                f"{uuid.uuid4()}.jpg"
            )

            final_image_path = os.path.join(
                RESULTS_FOLDER,
                unique_name
            )

            shutil.copy2(
                possible_image,
                final_image_path
            )

            result_image_url = (
                f"/results/predictions/{unique_name}"
            )

    return {

        "prediction": yolo_prediction,

        "confidence": round(
            yolo_confidence,
            2
        ),

        "detections": detections,

        "fake_region_count": (
            fake_region_count
        ),

        "true_region_count": (
            true_region_count
        ),

        "fake_region_confidence": round(
            max(fake_confidences) * 100,
            2
        ) if fake_confidences else 0.0,

        "true_region_confidence": round(
            max(true_confidences) * 100,
            2
        ) if true_confidences else 0.0,

        "result_image": result_image_url
    }


# ============================================================
# COMBINED PREDICTION
# ============================================================

def predict(image_path):

    # --------------------------------------------------------
    # RESNET50
    # --------------------------------------------------------

    resnet_result = predict_with_resnet(
        image_path
    )


    # --------------------------------------------------------
    # YOLO11
    # --------------------------------------------------------

    yolo_result = predict_with_yolo(
        image_path
    )


    # --------------------------------------------------------
    # FINAL CLASSIFICATION
    #
    # ResNet50 is the whole-image classifier.
    # YOLO11 provides suspicious-region evidence.
    # --------------------------------------------------------

    final_prediction = (
        resnet_result["prediction"]
    )

    final_confidence = (
        resnet_result["confidence"]
    )


    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    if (
        final_prediction == "Authentic"
        and yolo_result["fake_region_count"] > 0
    ):

        interpretation = (
            "Overall classification is Authentic, "
            "but YOLO11 detected a suspicious region "
            "for further review."
        )

    elif (
        final_prediction == "Authentic"
        and yolo_result["fake_region_count"] == 0
    ):

        interpretation = (
            "Overall classification is Authentic, "
            "and no suspicious YOLO11 region was detected."
        )

    elif (
        final_prediction == "Forged"
        and yolo_result["fake_region_count"] > 0
    ):

        interpretation = (
            "Overall classification is Forged, "
            "and YOLO11 also detected one or more "
            "suspicious regions."
        )

    else:

        interpretation = (
            "Overall classification is Forged, "
            "although YOLO11 did not detect a "
            "suspicious region."
        )


    # --------------------------------------------------------
    # FINAL RESPONSE
    # --------------------------------------------------------

    return {

        "prediction": final_prediction,

        "confidence": final_confidence,

        "resnet_prediction": (
            resnet_result["prediction"]
        ),

        "resnet_confidence": (
            resnet_result["confidence"]
        ),

        "forged_probability": (
            resnet_result["forged_probability"]
        ),

        "authentic_probability": (
            resnet_result["authentic_probability"]
        ),

        "yolo_prediction": (
            yolo_result["prediction"]
        ),

        "yolo_confidence": (
            yolo_result["confidence"]
        ),

        "detections": (
            yolo_result["detections"]
        ),

        "fake_region_count": (
            yolo_result["fake_region_count"]
        ),

        "true_region_count": (
            yolo_result["true_region_count"]
        ),

        "fake_region_confidence": (
            yolo_result["fake_region_confidence"]
        ),

        "true_region_confidence": (
            yolo_result["true_region_confidence"]
        ),

        "result_image": (
            yolo_result["result_image"]
        ),

        "interpretation": interpretation
    }

# ============================================================
# BACKWARD-COMPATIBILITY WRAPPER
# ============================================================

class CertificateModel:
    def predict(self, image_path):
        return predict(image_path)


certificate_model = CertificateModel()