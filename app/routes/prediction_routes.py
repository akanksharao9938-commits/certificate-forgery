from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.model_service import certificate_model
from app.services.image_analysis import analyze_pixel_inconsistency

import os
import uuid


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/api",
    tags=["Certificate Detection"]
)


# ============================================================
# UPLOAD FOLDER
# ============================================================

UPLOAD_FOLDER = r"D:\certificate-forgery-backend(2)\uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# ALLOWED FILE TYPES
# ============================================================

ALLOWED_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png"
}


# ============================================================
# PREDICT CERTIFICATE
# ============================================================

@router.post("/predict")
async def predict_certificate(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # CHECK FILE TYPE
    # --------------------------------------------------------

    if file.content_type not in ALLOWED_TYPES:

        raise HTTPException(
            status_code=400,
            detail=(
                "Only JPG and PNG certificate "
                "images are allowed."
            )
        )

    # --------------------------------------------------------
    # READ FILE
    # --------------------------------------------------------

    image_bytes = await file.read()

    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    # --------------------------------------------------------
    # CREATE UNIQUE FILE NAME
    # --------------------------------------------------------

    extension = os.path.splitext(
        file.filename
    )[1].lower()

    unique_filename = (
        f"{uuid.uuid4()}{extension}"
    )

    image_path = os.path.join(
        UPLOAD_FOLDER,
        unique_filename
    )

    # --------------------------------------------------------
    # SAVE TEMPORARY IMAGE
    # --------------------------------------------------------

    try:

        with open(
            image_path,
            "wb"
        ) as buffer:

            buffer.write(
                image_bytes
            )

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = certificate_model.predict(
            image_path
        )

        # ----------------------------------------------------
        # PIXEL ANALYSIS
        # ----------------------------------------------------

        pixel_analysis = (
            analyze_pixel_inconsistency(
                image_bytes
            )
        )

        # ----------------------------------------------------
        # FINAL RESPONSE
        # ----------------------------------------------------

        response = {

            "success": True,

            "filename": file.filename,

            # Overall result
            "prediction": prediction[
                "prediction"
            ],

            "confidence": prediction[
                "confidence"
            ],

            # ResNet50
            "classification": {

                "model": "ResNet50",

                "prediction": prediction[
                    "resnet_prediction"
                ],

                "confidence": prediction[
                    "resnet_confidence"
                ],

                "forged_probability": prediction[
                    "forged_probability"
                ],

                "authentic_probability": prediction[
                    "authentic_probability"
                ]
            },

            # YOLO11
            "region_detection": {

                "model": "YOLO11",

                "summary": prediction[
                    "yolo_prediction"
                ],

                "confidence": prediction[
                    "yolo_confidence"
                ],

                "detections": prediction[
                    "detections"
                ]
            },

            # Pixel analysis
            "pixel_analysis": pixel_analysis,

            # Message
            "message": (

                "Possible certificate forgery detected."
                if prediction["prediction"]
                == "Forged"

                else

                "Certificate appears authentic."
            )
        }

        return response

    # --------------------------------------------------------
    # MODEL FILE ERROR
    # --------------------------------------------------------

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )

    # --------------------------------------------------------
    # OTHER ERRORS
    # --------------------------------------------------------

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}"
        )

    # --------------------------------------------------------
    # DELETE TEMP IMAGE
    # --------------------------------------------------------

    finally:

        if os.path.exists(
            image_path
        ):

            os.remove(
                image_path
            )