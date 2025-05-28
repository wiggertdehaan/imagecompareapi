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

def resize_if_needed(img, max_dimension=1500):
    """Pas de afbeelding aan als deze te groot is"""
    height, width = img.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(img, (new_width, new_height))
    return img

def preprocess_image(img):
    try:
        # Resize als nodig
        img = resize_if_needed(img)
        
        # Converteer naar grijswaarden
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Pas histogram equalization toe voor betere contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Ruisreductie
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return gray
    except Exception as e:
        app.logger.error(f"Error in preprocess_image: {str(e)}")
        raise

def find_good_matches(des1, des2, ratio_thresh=0.75):
    # Gebruik BFMatcher met k=2 voor ratio test
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Pas Lowe's ratio test toe
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    
    return good_matches

def verify_geometric_consistency(kp1, kp2, good_matches, min_matches=10):
    if len(good_matches) < min_matches:
        return False, 0
    
    # Haal de coördinaten van de matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Bereken de homografie
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        return False, 0
    
    # Tel hoeveel inliers we hebben
    inliers = np.sum(mask)
    inlier_ratio = inliers / len(good_matches)
    
    return inlier_ratio > 0.5, inlier_ratio

@app.route('/')
def hello():
    return 'Image Compare API is running ✅'

@app.route('/compare', methods=['POST'])
def compare_images():
    try:
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': 'Beide afbeeldingen moeten worden meegegeven als image1 en image2'}), 400

        # Controleer bestandsgrootte
        if request.files['image1'].content_length and request.files['image1'].content_length > 10 * 1024 * 1024:  # 10MB
            return jsonify({'error': 'Afbeelding 1 is te groot (max 10MB)'}), 400
        if request.files['image2'].content_length and request.files['image2'].content_length > 10 * 1024 * 1024:  # 10MB
            return jsonify({'error': 'Afbeelding 2 is te groot (max 10MB)'}), 400

        tmp1 = tempfile.NamedTemporaryFile(delete=False)
        tmp2 = tempfile.NamedTemporaryFile(delete=False)
        
        try:
            request.files['image1'].save(tmp1.name)
            request.files['image2'].save(tmp2.name)

            img1 = cv2.imread(tmp1.name, cv2.IMREAD_COLOR)
            img2 = cv2.imread(tmp2.name, cv2.IMREAD_COLOR)

            if img1 is None or img2 is None:
                app.logger.error("Kon een van de afbeeldingen niet lezen")
                return jsonify({'error': 'Kon een van de afbeeldingen niet lezen'}), 400

            # Log afbeeldingsgrootte
            app.logger.info(f"Image 1 size: {img1.shape}, Image 2 size: {img2.shape}")

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

            # Pre-process de afbeeldingen
            processed_img1 = preprocess_image(right_half_img1)
            processed_img2 = preprocess_image(left_half_img2)

            # Gebruik ORB met meer features en betere parameters
            orb = cv2.ORB_create(
                nfeatures=2000,  # Verminderd van 3000 naar 2000 voor betere performance
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=31,
                firstLevel=0,
                WTA_K=2,
                patchSize=31,
                fastThreshold=20
            )
            
            kp1, des1 = orb.detectAndCompute(processed_img1, None)
            kp2, des2 = orb.detectAndCompute(processed_img2, None)

            if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
                app.logger.warning("Geen kenmerken gevonden in een of beide afbeeldingen")
                return jsonify({
                    "match": False,
                    "confidence": 0,
                    "remarks": "Geen kenmerken gevonden in een of beide afbeeldingen."
                })

            # Vind goede matches met ratio test
            good_matches = find_good_matches(des1, des2)
            
            # Verifieer geometrische consistentie
            is_geometrically_consistent, geometric_confidence = verify_geometric_consistency(kp1, kp2, good_matches)
            
            # Bereken de uiteindelijke confidence score
            match_ratio = len(good_matches) / min(len(des1), len(des2))
            final_confidence = int((match_ratio * 0.4 + geometric_confidence * 0.6) * 100)
            
            result = {
                "match": is_geometrically_consistent and final_confidence > 30,
                "confidence": final_confidence,
                "remarks": f"{len(good_matches)} goede matches gevonden. Geometrische consistentie: {int(geometric_confidence * 100)}%"
            }
            
            app.logger.info(f"Successfully processed images. Result: {result}")
            return jsonify(result)

        finally:
            # Cleanup
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
