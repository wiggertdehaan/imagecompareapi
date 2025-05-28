from flask import Flask, request, jsonify
import cv2
import numpy as np
import tempfile
import os
from typing import Tuple, Dict, Any
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

class ImageComparer:
    def __init__(self):
        # Verschillende feature detectors voor betere resultaten
        self.orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nLevels=8)
        self.sift = cv2.SIFT_create(nfeatures=2000)
        
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Voorbewerking van afbeelding voor betere feature detection"""
        # Histogram equalization voor betere contrast
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:,:,0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Lichte blur om ruis te verminderen
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        return img
    
    def detect_and_match_features(self, img1: np.ndarray, img2: np.ndarray, method='orb') -> Tuple[list, int, int]:
        """Feature detection en matching met verschillende algoritmes"""
        if method == 'sift':
            detector = self.sift
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            distance_threshold = 0.75
        else:  # ORB
            detector = self.orb
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            distance_threshold = 50
        
        # Feature detection
        kp1, des1 = detector.detectAndCompute(img1, None)
        kp2, des2 = detector.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None:
            return [], 0, 0
        
        # Matching
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Adaptieve threshold op basis van distance distributie
        if len(matches) > 10:
            distances = [m.distance for m in matches]
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            adaptive_threshold = mean_dist - 0.5 * std_dist
            if method == 'orb':
                distance_threshold = min(distance_threshold, max(20, adaptive_threshold))
        
        good_matches = [m for m in matches if m.distance < distance_threshold]
        
        return good_matches, len(matches), len(kp1) + len(kp2)
    
    def geometric_verification(self, img1: np.ndarray, img2: np.ndarray, matches: list) -> Dict[str, Any]:
        """Geometrische verificatie met homografie"""
        if len(matches) < 10:
            return {"valid": False, "inliers": 0, "homography_quality": 0}
        
        # Herdetecteer keypoints voor homografie
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)
        
        if len(matches) < 4:
            return {"valid": False, "inliers": 0, "homography_quality": 0}
        
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Vind homografie met RANSAC
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            return {"valid": False, "inliers": 0, "homography_quality": 0}
        
        inliers = np.sum(mask)
        inlier_ratio = inliers / len(matches)
        
        # Evalueer homografie kwaliteit
        det = np.linalg.det(M[:2, :2])
        homography_quality = 1 / (1 + abs(1 - abs(det)))  # Dichter bij 1 is beter
        
        return {
            "valid": inlier_ratio > 0.3 and inliers > 8,
            "inliers": int(inliers),
            "inlier_ratio": float(inlier_ratio),
            "homography_quality": float(homography_quality)
        }
    
    def template_matching_check(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Template matching als extra verificatie"""
        # Converteer naar grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Resize voor snelheid
        h1, w1 = gray1.shape
        h2, w2 = gray2.shape
        
        if h1 * w1 > h2 * w2:
            template = cv2.resize(gray2, (w1//4, h1//4))
            search_in = cv2.resize(gray1, (w1//2, h1//2))
        else:
            template = cv2.resize(gray1, (w2//4, h2//4))
            search_in = cv2.resize(gray2, (w2//2, h2//2))
        
        if template.shape[0] > search_in.shape[0] or template.shape[1] > search_in.shape[1]:
            return 0.0
        
        result = cv2.matchTemplate(search_in, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        return float(max_val)
    
    def analyze_overlap_region(self, img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
        """Analyseer potentiële overlap regio's aan de randen"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Neem randen van afbeeldingen
        edge_size = min(w1, w2, h1, h2) // 4
        
        # Rechterrand img1 vs linkerrand img2 (horizontale aansluiting)
        right_edge1 = img1[:, -edge_size:]
        left_edge2 = img2[:, :edge_size]
        
        # Onderrand img1 vs bovenrand img2 (verticale aansluiting)
        bottom_edge1 = img1[-edge_size:, :]
        top_edge2 = img2[:edge_size, :]
        
        # Bereken gelijkenis van randen
        def edge_similarity(edge1, edge2):
            if edge1.shape != edge2.shape:
                min_h = min(edge1.shape[0], edge2.shape[0])
                min_w = min(edge1.shape[1], edge2.shape[1])
                edge1 = cv2.resize(edge1, (min_w, min_h))
                edge2 = cv2.resize(edge2, (min_w, min_h))
            
            # Histogram vergelijking
            hist1 = cv2.calcHist([edge1], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([edge2], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        horizontal_sim = edge_similarity(right_edge1, left_edge2)
        vertical_sim = edge_similarity(bottom_edge1, top_edge2)
        
        return {
            "horizontal_similarity": horizontal_sim,
            "vertical_similarity": vertical_sim,
            "best_edge_match": max(horizontal_sim, vertical_sim)
        }

@app.route('/')
def hello():
    return 'Advanced Image Compare API is running ✅'

@app.route('/compare', methods=['POST'])
def compare_images():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Beide afbeeldingen moeten worden meegegeven als image1 en image2'}), 400

    # Optionele parameters
    use_sift = request.form.get('use_sift', 'false').lower() == 'true'
    detailed_analysis = request.form.get('detailed', 'false').lower() == 'true'
    
    tmp1 = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    
    try:
        request.files['image1'].save(tmp1.name)
        request.files['image2'].save(tmp2.name)

        img1 = cv2.imread(tmp1.name, cv2.IMREAD_COLOR)
        img2 = cv2.imread(tmp2.name, cv2.IMREAD_COLOR)

        if img1 is None or img2 is None:
            return jsonify({'error': 'Kon een van de afbeeldingen niet lezen'}), 400

        comparer = ImageComparer()
        
        # Intelligente resize op basis van afbeeldingsgrootte
        max_dimension = 800
        def smart_resize(img):
            h, w = img.shape[:2]
            if max(h, w) > max_dimension:
                scale = max_dimension / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return img
        
        img1 = smart_resize(img1)
        img2 = smart_resize(img2)
        
        # Preprocessing
        img1_processed = comparer.preprocess_image(img1)
        img2_processed = comparer.preprocess_image(img2)
        
        # Feature matching met meerdere methodes
        methods_to_try = ['sift', 'orb'] if use_sift else ['orb']
        best_confidence = 0
        best_result = None
        
        for method in methods_to_try:
            good_matches, total_matches, total_keypoints = comparer.detect_and_match_features(
                img1_processed, img2_processed, method
            )
            
            if total_matches == 0:
                continue
                
            # Basis confidence
            match_ratio = len(good_matches) / total_matches
            confidence = min(100, int(match_ratio * 100))
            
            # Geometrische verificatie
            geom_result = comparer.geometric_verification(img1_processed, img2_processed, good_matches)
            
            # Template matching
            template_score = comparer.template_matching_check(img1, img2)
            
            # Edge analysis voor supermarkt foto's
            edge_analysis = comparer.analyze_overlap_region(img1, img2)
            
            # Gecombineerde score
            combined_confidence = (
                confidence * 0.4 +
                (geom_result["inlier_ratio"] * 100) * 0.3 +
                (template_score * 100) * 0.15 +
                (edge_analysis["best_edge_match"] * 100) * 0.15
            )
            
            if combined_confidence > best_confidence:
                best_confidence = combined_confidence
                best_result = {
                    "method": method,
                    "match": combined_confidence > 35,  # Aangepaste threshold
                    "confidence": int(combined_confidence),
                    "feature_matches": len(good_matches),
                    "total_matches": total_matches,
                    "geometric_verification": geom_result,
                    "template_score": round(template_score, 3),
                    "edge_analysis": edge_analysis if detailed_analysis else None,
                    "remarks": f"{len(good_matches)} goede matches van {total_matches} totaal. "
                              f"Geometrische verificatie: {geom_result['valid']}"
                }
        
        if best_result is None:
            return jsonify({
                "match": False,
                "confidence": 0,
                "error": "Geen features gevonden in een of beide afbeeldingen"
            })
        
        # Voeg extra context toe voor supermarkt gebruik
        if best_result["edge_analysis"]:
            if best_result["edge_analysis"]["horizontal_similarity"] > 0.7:
                best_result["remarks"] += " Sterke horizontale aansluiting gedetecteerd."
            elif best_result["edge_analysis"]["vertical_similarity"] > 0.7:
                best_result["remarks"] += " Sterke verticale aansluiting gedetecteerd."
        
        return jsonify(best_result)
        
    except Exception as e:
        logging.error(f"Error in image comparison: {str(e)}")
        return jsonify({'error': f'Er is een fout opgetreden: {str(e)}'}), 500
        
    finally:
        try:
            os.unlink(tmp1.name)
            os.unlink(tmp2.name)
        except:
            pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)