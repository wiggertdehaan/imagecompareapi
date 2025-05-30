from flask import Flask, request, jsonify
import cv2
import numpy as np
import tempfile
import os
import logging
from logging.handlers import RotatingFileHandler
import traceback

app = Flask(__name__)

# Configureer logging
if not os.path.exists('logs'):
    os.makedirs('logs')
file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('Image Compare API startup')

def compare_images_v3(img1, img2):
    """Vergelijk twee afbeeldingen met de exacte schaalstrategie van original.py.
       - Geen smart_resize.
       - img2 wordt direct geschaald naar de originele dimensies van img1.
       - ORB(8000), BFMatcher(crossCheck=True).
       - Good match distance < 50.
       - Match als confidence > 30%.
    """
    # Stap 1: Bepaal de dimensies van de originele img1
    h, w = img1.shape[:2]
    
    # Stap 2: Schaal img2 naar de exacte dimensies van de originele img1
    img1_final = img1 
    img2_final = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)

    # Gebruik ORB met nfeatures=8000
    orb = cv2.ORB_create(nfeatures=8000)
    kp1, des1 = orb.detectAndCompute(img1_final, None)
    kp2, des2 = orb.detectAndCompute(img2_final, None)

    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return False, 0, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    matches = sorted(matches, key=lambda x: x.distance)
    
    good_matches = [m for m in matches if m.distance < 50]
    
    confidence = 0
    if matches and len(matches) > 0:
        confidence = min(100, int((len(good_matches) / len(matches)) * 100))
    
    is_match = confidence > 30

    return is_match, confidence, len(good_matches)

@app.route('/')
def hello():
    return 'Image Compare API is running ✅'

@app.route('/compare', methods=['POST'])
def compare_images():
    try:
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': 'Beide afbeeldingen moeten worden meegegeven als image1 en image2'}), 400

        if request.files['image1'].content_length and request.files['image1'].content_length > 10 * 1024 * 1024:  # 10MB
            return jsonify({'error': 'Afbeelding 1 is te groot (max 10MB)'}), 400
        if request.files['image2'].content_length and request.files['image2'].content_length > 10 * 1024 * 1024:  # 10MB
            return jsonify({'error': 'Afbeelding 2 is te groot (max 10MB)'}), 400

        tmp1 = tempfile.NamedTemporaryFile(delete=False)
        tmp2 = tempfile.NamedTemporaryFile(delete=False)
        
        try:
            request.files['image1'].save(tmp1.name)
            request.files['image2'].save(tmp2.name)

            img1_read = cv2.imread(tmp1.name, cv2.IMREAD_COLOR)
            img2_read = cv2.imread(tmp2.name, cv2.IMREAD_COLOR)

            if img1_read is None or img2_read is None:
                app.logger.error("Kon een van de afbeeldingen niet lezen")
                return jsonify({'error': 'Kon een van de afbeeldingen niet lezen'}), 400

            app.logger.info(f"Image 1 (input) size: {img1_read.shape}, Image 2 (input) size: {img2_read.shape}")

            match_v3, confidence_v3, num_good_matches = compare_images_v3(img1_read, img2_read)
            
            result = {
                "match": bool(match_v3),
                "confidence": int(confidence_v3),
                "remarks": f"{num_good_matches} goede matches gevonden. Confidence: {int(confidence_v3)}%"
            }
            
            app.logger.info(f"Successfully processed images. Result: {result}")
            return jsonify(result)

        finally:
            try:
                os.unlink(tmp1.name)
                os.unlink(tmp2.name)
            except Exception as e:
                app.logger.error(f"Error during cleanup: {str(e)}")

    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'error': 'Er is een onverwachte fout opgetreden bij het verwerken van de afbeeldingen',
            'details': str(e) if app.debug else None
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
