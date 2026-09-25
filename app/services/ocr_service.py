import pytesseract
from pytesseract import Output
from PIL import Image


# Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = (
    r"D:\tesseract\tesseract.exe"
)


def extract_ocr_data(image_path):
    """
    Extract words, bounding boxes, and OCR confidence
    from a certificate/document image.
    """

    image = Image.open(image_path)

    data = pytesseract.image_to_data(
        image,
        output_type=Output.DICT
    )

    words = []

    for i in range(len(data["text"])):

        text = data["text"][i].strip()

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            confidence = -1

        # Ignore empty OCR results
        if not text:
            continue

        # Ignore invalid confidence values
        if confidence < 0:
            continue

        x = int(data["left"][i])
        y = int(data["top"][i])
        width = int(data["width"][i])
        height = int(data["height"][i])

        words.append({
            "text": text,
            "confidence": round(confidence, 2),
            "x": x,
            "y": y,
            "width": width,
            "height": height
        })

    return words