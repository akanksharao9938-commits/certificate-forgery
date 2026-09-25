import cv2
import numpy as np

from sklearn.preprocessing import StandardScaler

from app.services.ocr_service import extract_ocr_data
from app.services.feature_service import extract_spatial_features
from app.services.lof_service import detect_lof_anomalies


def analyze_text_anomalies(image_path):
    """
    Perform OCR-based typographical and spatial anomaly detection.

    Pipeline:

        Certificate image
                ↓
              OCR
                ↓
        Text + bounding boxes
                ↓
        Spatial features
                ↓
      Spatial-density features
                ↓
       StandardScaler
                ↓
              LOF
                ↓
        Local anomalies
    """


    # ========================================================
    # READ IMAGE
    # ========================================================

    image = cv2.imread(
        image_path
    )

    if image is None:
        raise ValueError(
            "Unable to read image."
        )


    image_height, image_width = (
        image.shape[:2]
    )


    # ========================================================
    # OCR
    # ========================================================

    words = extract_ocr_data(
        image_path
    )


    # ========================================================
    # NO TEXT DETECTED
    # ========================================================

    if not words:

        return {

            "ocr_word_count": 0,

            "anomaly_count": 0,

            "anomaly_rate_percent": 0,

            "anomalies": [],

            "spatial_density_features": {
                "enabled": True,
                "feature_count": 12,
                "features": [
                    "normalized_x",
                    "normalized_y",
                    "normalized_width",
                    "normalized_height",
                    "aspect_ratio",
                    "ocr_confidence",
                    "nearest_neighbor_distance",
                    "average_neighbor_distance",
                    "neighbor_count",
                    "local_spatial_density",
                    "normalized_text_area",
                    "relative_vertical_position"
                ]
            },

            "message":
                "No readable text detected."
        }


    # ========================================================
    # EXTRACT SPATIAL + DENSITY FEATURES
    # ========================================================

    features = extract_spatial_features(

        words,

        image_width,

        image_height,

        density_radius=0.12
    )


    # ========================================================
    # STANDARDIZE FEATURES
    # ========================================================
    #
    # Different features have different scales.
    #
    # Example:
    #     neighbor_count may be 10
    #     aspect_ratio may be 3.5
    #     normalized_x may be 0.4
    #
    # StandardScaler makes the features comparable before
    # applying LOF.
    # ========================================================

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        features
    )


    # ========================================================
    # LOF
    # ========================================================

    lof_result = detect_lof_anomalies(
        scaled_features
    )


    labels = lof_result[
        "labels"
    ]

    scores = lof_result[
        "scores"
    ]


    # ========================================================
    # COLLECT ANOMALIES
    # ========================================================

    anomalies = []


    for word, label, score in zip(
        words,
        labels,
        scores
    ):

        if label == -1:

            anomalies.append({

                "text":
                    word["text"],

                "x":
                    word["x"],

                "y":
                    word["y"],

                "width":
                    word["width"],

                "height":
                    word["height"],

                "ocr_confidence":
                    word["confidence"],

                "lof_score":
                    round(
                        float(score),
                        3
                    )
            })


    # ========================================================
    # ANOMALY STATISTICS
    # ========================================================

    anomaly_count = len(
        anomalies
    )


    anomaly_rate = (
        anomaly_count
        / len(words)
    ) * 100


    # ========================================================
    # MESSAGE
    # ========================================================

    if anomaly_count > 0:

        message = (
            "Local typographical and "
            "spatial-density anomalies detected."
        )

    else:

        message = (
            "No significant local "
            "typographical or spatial anomalies detected."
        )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {

        "ocr_word_count":
            len(words),

        "anomaly_count":
            anomaly_count,

        "anomaly_rate_percent":
            round(
                anomaly_rate,
                2
            ),

        "anomalies":
            anomalies,

        "spatial_density_features": {

            "enabled":
                True,

            "feature_count":
                12,

            "density_radius":
                0.12,

            "features": [

                "normalized_x",

                "normalized_y",

                "normalized_width",

                "normalized_height",

                "aspect_ratio",

                "ocr_confidence",

                "nearest_neighbor_distance",

                "average_neighbor_distance",

                "neighbor_count",

                "local_spatial_density",

                "normalized_text_area",

                "relative_vertical_position"
            ]
        },

        "message":
            message
    }