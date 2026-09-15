from ultralytics import YOLO
import tensorflow as tf
import numpy as np
import os


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
    "results"
)

os.makedirs(RESULTS_FOLDER, exist_ok=True)


# ============================================================
# CERTIFICATE MODEL
# ============================================================

class CertificateModel:

    def __init__(self):
        self.yolo_model = None
        self.resnet_model = None

    # ========================================================
    # LOAD YOLO
    # ========================================================

    def load_yolo_model(self):

        if not os.path.exists(YOLO_MODEL_PATH):
            raise FileNotFoundError(
                f"YOLO model not found at:\n{YOLO_MODEL_PATH}"
            )

        print("Loading YOLO11 model...")

        self.yolo_model = YOLO(
            YOLO_MODEL_PATH
        )

        print("YOLO11 model loaded successfully.")

        return self.yolo_model

    # ========================================================
    # LOAD RESNET50
    # ========================================================

    def load_resnet_model(self):

        if not os.path.exists(RESNET_MODEL_PATH):
            raise FileNotFoundError(
                f"ResNet50 model not found at:\n{RESNET_MODEL_PATH}"
            )

        print("Loading improved ResNet50 model...")

        self.resnet_model = tf.keras.models.load_model(
            RESNET_MODEL_PATH
        )

        print("Improved ResNet50 model loaded successfully.")

        return self.resnet_model

    # ========================================================
    # YOLO PREDICTION
    # ========================================================

    def predict_yolo(self, image_path):

        if self.yolo_model is None:
            self.load_yolo_model()

        results = self.yolo_model.predict(
            source=image_path,
            imgsz=416,
            conf=0.25,
            save=True,
            project=RESULTS_FOLDER,
            name="predictions",
            exist_ok=True,
            verbose=False
        )

        result = results[0]

        detections = []

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                class_name = self.yolo_model.names[
                    class_id
                ]

                x1, y1, x2, y2 = box.xyxy[
                    0
                ].tolist()

                detections.append({

                    "class": class_name,

                    "confidence": round(
                        confidence * 100,
                        2
                    ),

                    "bounding_box": {

                        "x1": round(
                            x1,
                            2
                        ),

                        "y1": round(
                            y1,
                            2
                        ),

                        "x2": round(
                            x2,
                            2
                        ),

                        "y2": round(
                            y2,
                            2
                        )
                    }
                })

        return detections

    # ========================================================
    # RESNET50 PREDICTION
    # ========================================================

    def predict_resnet(self, image_path):

        if self.resnet_model is None:
            self.load_resnet_model()

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

        prediction = self.resnet_model.predict(
            image_array,
            verbose=0
        )

        forged_probability = float(
            prediction[0][0]
        )

        authentic_probability = (
            1 - forged_probability
        )

        if forged_probability >= 0.5:

            classification = "Forged"

            confidence = (
                forged_probability * 100
            )

        else:

            classification = "Authentic"

            confidence = (
                authentic_probability * 100
            )

        return {

            "prediction": classification,

            "confidence": round(
                confidence,
                2
            ),

            "forged_probability": round(
                forged_probability * 100,
                2
            ),

            "authentic_probability": round(
                authentic_probability * 100,
                2
            )
        }

    # ========================================================
    # FINAL PREDICTION
    # ========================================================

    def predict(self, image_path):

        # ----------------------------------------------------
        # 1. YOLO
        # ----------------------------------------------------

        detections = self.predict_yolo(
            image_path
        )

        fake_detections = [

            detection

            for detection in detections

            if detection["class"].lower()
            == "fake"

        ]

        # ----------------------------------------------------
        # YOLO IS SUPPORTING EVIDENCE
        # ----------------------------------------------------

        if fake_detections:

            best_fake = max(
                fake_detections,
                key=lambda x: x["confidence"]
            )

            yolo_prediction = "Suspicious regions detected"

            yolo_confidence = best_fake[
                "confidence"
            ]

        else:

            yolo_prediction = "No fake region detected"

            yolo_confidence = 0

        # ----------------------------------------------------
        # 2. RESNET50
        # ----------------------------------------------------

        resnet_result = self.predict_resnet(
            image_path
        )

        # ----------------------------------------------------
        # 3. FINAL DECISION
        # ----------------------------------------------------

        final_prediction = resnet_result[
            "prediction"
        ]

        final_confidence = resnet_result[
            "confidence"
        ]

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        return {

            "prediction": final_prediction,

            "confidence": final_confidence,

            "resnet_prediction": resnet_result[
                "prediction"
            ],

            "resnet_confidence": resnet_result[
                "confidence"
            ],

            "forged_probability": resnet_result[
                "forged_probability"
            ],

            "authentic_probability": resnet_result[
                "authentic_probability"
            ],

            "yolo_prediction": yolo_prediction,

            "yolo_confidence": yolo_confidence,

            "detections": detections,

            "result_image": os.path.join(
                RESULTS_FOLDER,
                "predictions"
            )
        }


# ============================================================
# GLOBAL MODEL INSTANCE
# ============================================================

certificate_model = CertificateModel()