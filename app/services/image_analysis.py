import cv2
import numpy as np

def analyze_pixel_inconsistency(image_bytes):
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image file.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200)
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_pixels = int(np.count_nonzero(edges))
    edge_density = (edge_pixels / total_pixels) * 100

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    difference = cv2.absdiff(gray, blur)
    noise_score = float(np.mean(difference))

    return {
        "edge_density_percent": round(float(edge_density), 2),
        "pixel_noise_score": round(noise_score, 2)
    }
