from app.services.ocr_service import extract_ocr_data


IMAGE_PATH = r"D:\\certificate-forgery-backend(2)\\runs\\detect\\predict\\f2_png.rf.0d3227aadf72ecadd8a3e773a96f5015.jpg"

words = extract_ocr_data(IMAGE_PATH)

print("\nOCR RESULTS")
print("=" * 50)

print(f"Words detected: {len(words)}")

for word in words:
    print(word)