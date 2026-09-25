# Optical Character Forgery Anomaly Detection

## Pattern Recognition Project

### Detection of Textual and Spatial Anomalies in Academic Certificate Images Using OCR and Local Outlier Factor

---

## 📌 Project Overview

Academic certificates contain important textual information such as student names, degree names, registration numbers, dates, institution names, and other academic details.

Modification of certificate text using image-editing tools can be difficult to identify through manual inspection.

This project proposes an **Optical Character Forgery Anomaly Detection** system that uses **Optical Character Recognition (OCR)** and **Pattern Recognition** techniques to identify unusual textual and spatial patterns in academic certificate images.

The system extracts text from a certificate image using **Tesseract OCR**. Along with the text, it extracts the position, size, shape, and OCR confidence of each detected text region.

These properties are converted into numerical features and analyzed using the **Local Outlier Factor (LOF)** algorithm.

The purpose of the system is to identify **potentially anomalous text regions that require further investigation**.

> An anomaly detected by the system does not automatically prove that a certificate is forged. It is a supporting indication for further verification.

---

## 🎯 Objectives

The main objectives of this project are:

- Extract textual information from certificate images using OCR.
- Obtain bounding-box coordinates for detected text regions.
- Extract spatial and OCR-related features.
- Analyze relationships between neighboring text regions.
- Apply Pattern Recognition techniques to certificate text regions.
- Use Local Outlier Factor for local anomaly detection.
- Identify potentially unusual text regions.
- Calculate the anomaly rate.
- Provide understandable anomaly information to the user.

---

## 🔍 Problem Statement

To develop an OCR-based anomaly detection system that extracts textual and spatial features from academic certificate images and identifies potentially abnormal text regions using the Local Outlier Factor algorithm.

---

## 🧠 Pattern Recognition Approach

Pattern Recognition is the main concept of this project.

Each detected OCR text region is represented using a feature vector describing its visual and spatial characteristics.

The system learns the local pattern of the text regions and identifies regions that behave differently from their neighboring regions.

The overall pattern recognition process is:

```text
Certificate Image
       ↓
     OCR
       ↓
Text + Bounding Boxes + Confidence
       ↓
Feature Extraction
       ↓
Feature Standardization
       ↓
Local Outlier Factor
       ↓
Normal / Potential Anomaly
       ↓
Anomaly Report


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
