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

def enhance_contrast(img):
    """Verbeter contrast met CLAHE op verschillende kanalen"""
    if len(img.shape) == 2:  # Als het al grijswaarden is
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        return clahe.apply(img)
    else:  # Voor kleurenafbeeldingen
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def detect_edges(img):
    """Edge detection met Canny en Sobel"""
    # Gaussian blur voor ruisreductie
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Canny edge detection
    edges_canny = cv2.Canny(blurred, 50, 150)
    
    # Sobel edge detection
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    edges_sobel = np.sqrt(sobelx**2 + sobely**2)
    edges_sobel = np.uint8(edges_sobel)
    
    # Combineer beide edge maps
    edges = cv2.addWeighted(edges_canny, 0.7, edges_sobel, 0.3, 0)
    return edges

def preprocess_image(img):
    try:
        # Resize als nodig
        img = resize_if_needed(img)
        
        # Verbeter contrast
        img = enhance_contrast(img)
        
        # Converteer naar grijswaarden
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = detect_edges(gray)
        
        # Combineer grijswaarden en edges
        combined = cv2.addWeighted(gray, 0.7, edges, 0.3, 0)
        
        return combined
    except Exception as e:
        app.logger.error(f"Error in preprocess_image: {str(e)}")
        raise

def analyze_match_distribution(kp1, kp2, good_matches, img_shape):
    """Analyseer de verdeling van matches over de afbeelding"""
    if not good_matches:
        return 0.0
    
    # Bereken de coördinaten van matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    # Bereken de afstanden tussen corresponderende punten
    distances = np.linalg.norm(src_pts - dst_pts, axis=1)
    
    # Bereken de gemiddelde afstand
    avg_distance = np.mean(distances)
    
    # Bereken de standaarddeviatie van de afstanden
    std_distance = np.std(distances)
    
    # Bereken de overlap score (lager is beter)
    overlap_score = avg_distance / (img_shape[0] * 0.1)  # Normaliseer op basis van afbeeldingshoogte
    
    # Bereken de consistentie score (lager is beter)
    consistency_score = std_distance / avg_distance if avg_distance > 0 else 1.0
    
    return 1.0 - min(1.0, (overlap_score + consistency_score) / 2.0)

def find_good_matches(des1, des2, ratio_thresh=0.7):  # Aangepaste ratio threshold
    """Verbeterde feature matching met ratio test en symmetrie check"""
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return []
    
    # Forward matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches_forward = bf.knnMatch(des1, des2, k=2)
    
    # Backward matching
    matches_backward = bf.knnMatch(des2, des1, k=2)
    
    # Pas ratio test toe op beide richtingen
    good_matches_forward = []
    for m, n in matches_forward:
        if m.distance < ratio_thresh * n.distance:
            good_matches_forward.append(m)
    
    good_matches_backward = []
    for m, n in matches_backward:
        if m.distance < ratio_thresh * n.distance:
            good_matches_backward.append(m)
    
    # Symmetrie check
    good_matches = []
    for match in good_matches_forward:
        # Zoek de corresponderende backward match
        for back_match in good_matches_backward:
            if match.queryIdx == back_match.trainIdx and match.trainIdx == back_match.queryIdx:
                good_matches.append(match)
                break
    
    return good_matches

def verify_geometric_consistency(kp1, kp2, good_matches, min_matches=8):  # Verminderd minimum matches
    if len(good_matches) < min_matches:
        return False, 0
    
    # Haal de coördinaten van de matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Bereken de homografie met RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)  # Aangepaste RANSAC threshold
    
    if H is None:
        return False, 0
    
    # Tel hoeveel inliers we hebben
    inliers = np.sum(mask)
    inlier_ratio = inliers / len(good_matches)
    
    # Bereken de overlap score
    overlap_score = analyze_match_distribution(kp1, kp2, good_matches, 
                                             (max(kp1[0].size[1], kp2[0].size[1]),
                                              max(kp1[0].size[0], kp2[0].size[0])))
    
    # Combineer inlier ratio en overlap score
    final_score = (inlier_ratio * 0.6 + overlap_score * 0.4)
    
    return final_score > 0.3, final_score  # Aangepaste drempelwaarde

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

            # Gebruik ORB met verbeterde parameters
            orb = cv2.ORB_create(
                nfeatures=3000,  # Verhoogd aantal features
                scaleFactor=1.15,  # Fijnere schaal
                nlevels=12,  # Meer niveaus voor betere schaal-invariantie
                edgeThreshold=31,
                firstLevel=0,
                WTA_K=2,
                patchSize=31,
                fastThreshold=15  # Verlaagd voor meer features
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
            final_confidence = int((match_ratio * 0.3 + geometric_confidence * 0.7) * 100)  # Aangepaste gewichten
            
            # Converteer NumPy types naar Python native types
            result = {
                "match": bool(is_geometrically_consistent and final_confidence > 25),  # Verlaagde drempelwaarde
                "confidence": int(final_confidence),
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
