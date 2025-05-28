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

        # Bereken de helft van de breedte voor beide afbeeldingen
        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]
        
        # Neem de rechterhelft van de eerste afbeelding
        right_half_img1 = img1[:, width1//2:]
        # Neem de linkerhelft van de tweede afbeelding
        left_half_img2 = img2[:, :width2//2]

        # Zorg ervoor dat beide helften dezelfde hoogte hebben
        min_height = min(right_half_img1.shape[0], left_half_img2.shape[0])
        right_half_img1 = right_half_img1[:min_height, :]
        left_half_img2 = left_half_img2[:min_height, :]

        # Gebruik de halve afbeeldingen voor de vergelijking
        orb = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = orb.detectAndCompute(right_half_img1, None)
        kp2, des2 = orb.detectAndCompute(left_half_img2, None)

        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            return jsonify({
                "match": False,
                "confidence": 0,
                "remarks": "Geen kenmerken gevonden in een of beide afbeeldingen."
            })

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
