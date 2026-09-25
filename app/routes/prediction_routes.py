from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException
)

from app.services.model_service import certificate_model
from app.services.image_analysis import (
    analyze_pixel_inconsistency
)

from app.services.text_anomaly_service import (
    analyze_text_anomalies
)

import os
import uuid


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(

    prefix="/api",

    tags=[
        "Certificate Detection"
    ]

)


# ============================================================
# FOLDERS
# ============================================================

BASE_DIR = (
    r"D:\certificate-forgery-backend(2)"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

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
# PREDICT ENDPOINT
# ============================================================

@router.post(
    "/predict"
)
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
                "Only JPG and PNG "
                "certificate images are allowed."
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
    # CREATE TEMPORARY FILE
    # --------------------------------------------------------

    extension = os.path.splitext(
        file.filename or ""
    )[1].lower()


    if extension not in [
        ".jpg",
        ".jpeg",
        ".png"
    ]:

        extension = ".jpg"


    unique_filename = (
        f"{uuid.uuid4()}{extension}"
    )


    image_path = os.path.join(
        UPLOAD_FOLDER,
        unique_filename
    )


    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    try:

        # Save uploaded certificate

        with open(
            image_path,
            "wb"
        ) as buffer:

            buffer.write(
                image_bytes
            )


        # ----------------------------------------------------
        # AI MODEL ANALYSIS
        # ----------------------------------------------------

        prediction = (
            certificate_model.predict(
                image_path
            )
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
        # OCR + LOF
        # ----------------------------------------------------

        text_analysis = (
            analyze_text_anomalies(
                image_path
            )
        )


        # ----------------------------------------------------
        # FINAL API RESPONSE
        # ----------------------------------------------------

        response = {

            "success": True,

            "filename": file.filename,

            # -----------------------------------------------
            # FINAL RESNET CLASSIFICATION
            # -----------------------------------------------

            "prediction": (
                prediction["prediction"]
            ),

            "confidence": (
                prediction["confidence"]
            ),

            # -----------------------------------------------
            # RESNET50
            # -----------------------------------------------

            "classification": {

                "model": "ResNet50",

                "prediction": (
                    prediction[
                        "resnet_prediction"
                    ]
                ),

                "confidence": (
                    prediction[
                        "resnet_confidence"
                    ]
                ),

                "forged_probability": (
                    prediction[
                        "forged_probability"
                    ]
                ),

                "authentic_probability": (
                    prediction[
                        "authentic_probability"
                    ]
                )

            },


            # -----------------------------------------------
            # YOLO11
            # -----------------------------------------------

            "region_detection": {

                "model": "YOLO11",

                "summary": (
                    prediction[
                        "yolo_prediction"
                    ]
                ),

                "confidence": (
                    prediction[
                        "yolo_confidence"
                    ]
                ),

                "fake_region_count": (
                    prediction[
                        "fake_region_count"
                    ]
                ),

                "true_region_count": (
                    prediction[
                        "true_region_count"
                    ]
                ),

                "fake_region_confidence": (
                    prediction[
                        "fake_region_confidence"
                    ]
                ),

                "true_region_confidence": (
                    prediction[
                        "true_region_confidence"
                    ]
                ),

                "detections": (
                    prediction[
                        "detections"
                    ]
                )

            },


            # -----------------------------------------------
            # PIXEL ANALYSIS
            # -----------------------------------------------

            "pixel_analysis": (
                pixel_analysis
            ),


            # -----------------------------------------------
            # OCR + LOF
            # -----------------------------------------------

            "text_analysis": (
                text_analysis
            ),


            # -----------------------------------------------
            # RESULT IMAGE
            # -----------------------------------------------

            "result_image": (
                prediction[
                    "result_image"
                ]
            ),


            # -----------------------------------------------
            # INTERPRETATION
            # -----------------------------------------------

            "interpretation": (
                prediction[
                    "interpretation"
                ]
            ),


            # -----------------------------------------------
            # MESSAGE
            # -----------------------------------------------

            "message": (

                "Possible certificate "
                "forgery detected."

                if prediction[
                    "prediction"
                ] == "Forged"

                else

                "Certificate appears authentic."

            )

        }


        return response


    # --------------------------------------------------------
    # FILE NOT FOUND
    # --------------------------------------------------------

    except FileNotFoundError as exc:

        raise HTTPException(

            status_code=503,

            detail=str(exc)

        )


    # --------------------------------------------------------
    # OTHER ERROR
    # --------------------------------------------------------

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=(
                f"Prediction failed: {exc}"
            )

        )


    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    finally:

        if os.path.exists(
            image_path
        ):

            try:

                os.remove(
                    image_path
                )

            except Exception:

                pass