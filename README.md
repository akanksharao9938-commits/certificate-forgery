# CNN-Based Document Forgery Detection

Backend for the IBM Deep Learning project:
"Develop a convolutional neural network with residual connections to identify tampered regions and pixel inconsistencies in scanned academic certificates."

## Technology
- FastAPI
- TensorFlow / Keras
- ResNet50 (CNN with residual connections)
- OpenCV
- Pillow
- NumPy

## Folder structure

certificate-forgery-backend/
├── app/
│   ├── main.py
│   ├── routes/
│   │   └── prediction_routes.py
│   ├── services/
│   │   ├── model_service.py
│   │   └── image_analysis.py
│   └── utils/
│       └── preprocessing.py
├── dataset/
│   ├── train/
│   │   ├── authentic/
│   │   └── forged/
│   └── test/
│       ├── authentic/
│       └── forged/
├── model/
├── uploads/
├── results/
├── train_model.py
├── requirements.txt
└── README.md

## Setup on Windows / D drive

Open PowerShell:

D:
cd D:\certificate-forgery-backend

Create environment:

python -m venv venv

Activate:

.\venv\Scripts\Activate.ps1

Install:

python -m pip install -r requirements.txt

## Dataset

Put genuine images in:
dataset\train\authentic
dataset\test\authentic

Put forged/tampered images in:
dataset\train\forged
dataset\test\forged

Keep the classes reasonably balanced.

## Train

python train_model.py

After successful training:
model\certificate_model.keras

## Run API

uvicorn app.main:app --reload

Open:
http://127.0.0.1:8000

Swagger:
http://127.0.0.1:8000/docs

## API

POST /api/predict

Upload a JPG/PNG certificate.

The response contains:
- prediction
- confidence
- raw score
- pixel noise score
- edge density

## Important research note

The current model performs image-level authentic/forged classification and provides supporting pixel-analysis metrics. It does NOT by itself prove that a document is genuine or forged, and pixel-analysis values should not be interpreted as a legal/forensic conclusion.

For the full project, add a tampered-region localization module such as Grad-CAM or a segmentation model trained with tampering masks.
