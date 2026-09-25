from app.services.text_anomaly_service import analyze_text_anomalies


IMAGE_PATH = r"D:\\certificate-forgery-backend(2)\\runs\\detect\\predict\\f2_png.rf.0d3227aadf72ecadd8a3e773a96f5015.jpg"



result = analyze_text_anomalies(IMAGE_PATH)


print("\nOCR + LOF ANALYSIS")
print("=" * 60)

print(f"OCR words: {result['ocr_word_count']}")
print(f"Anomalies: {result['anomaly_count']}")
print(f"Anomaly rate: {result['anomaly_rate_percent']}%")

print("\nAnomalous regions:")

for anomaly in result["anomalies"]:
    print(anomaly)