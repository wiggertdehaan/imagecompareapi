from flask import Flask, request, jsonify
import cv2
import numpy as np
import tempfile
import os

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Image Compare API is running ✅'

@app.route('/compare', methods=['POST'])
def compare_images():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Beide afbeeldingen moeten worden meegegeven als image1 en image2'}), 400

    tmp1 = tempfile.NamedTemporaryFile(delete=False)
    tmp2 = tempfile.NamedTemporaryFile(delete=False)
    request.files['image1'].save(tmp1.name)
    request.files['image2'].save(tmp2.name)

    try:
        img1 = cv2.imread(tmp1.name, cv2.IMREAD_COLOR)
        img2 = cv2.imread(tmp2.name, cv2.IMREAD_COLOR)

        if img1 is None or img2 is None:
            return jsonify({'error': 'Kon een van de afbeeldingen niet lezen'}), 400

        scale_percent = 100
        width = int(img1.shape[1] * scale_percent / 100)
        height = int(img1.shape[0] * scale_percent / 100)
        dim = (width, height)
        img1 = cv2.resize(img1, dim)
        img2 = cv2.resize(img2, dim)

        orb = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = [m for m in matches if m.distance < 50]
        confidence = min(100, int((len(good_matches) / len(matches)) * 100)) if matches else 0

        result = {
            "match": confidence > 30,
            "confidence": confidence,
            "remarks": f"{len(good_matches)} goede matches gevonden van totaal {len(matches)}."
        }
        return jsonify(result)
    finally:
        os.unlink(tmp1.name)
        os.unlink(tmp2.name)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
