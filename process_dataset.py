import requests
import cv2
import numpy as np
import tempfile
import os
import hashlib
from urllib.parse import urlparse
from app import preprocess_image, find_good_matches, verify_geometric_consistency, compare_images_v3

def get_image_path(url):
    """Genereer een uniek pad voor een afbeelding op basis van de URL"""
    # Haal het bestandspad uit de URL
    parsed = urlparse(url)
    path = parsed.path
    
    # Maak een hash van de URL voor unieke identificatie
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    
    # Maak de images directory als deze nog niet bestaat
    if not os.path.exists('images'):
        os.makedirs('images')
    
    # Genereer het bestandspad
    filename = f"{url_hash}_{os.path.basename(path)}"
    return os.path.join('images', filename)

def download_image(url):
    """Download een afbeelding van de gegeven URL en sla deze lokaal op"""
    image_path = get_image_path(url)
    
    # Als de afbeelding al bestaat, gebruik deze
    if os.path.exists(image_path):
        print(f"Gebruik lokale kopie: {image_path}")
        img = cv2.imread(image_path)
        if img is not None:
            return img
    
    # Download de afbeelding als deze nog niet lokaal bestaat
    print(f"Download: {url}")
    response = requests.get(url + "?size=original")
    if response.status_code != 200:
        raise Exception(f"Kon afbeelding niet downloaden: {response.status_code}")
    
    # Sla de afbeelding op
    with open(image_path, 'wb') as f:
        f.write(response.content)
    
    # Lees met OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("Kon afbeelding niet lezen na download")
    return img

def process_dataset():
    """Verwerk de dataset en voeg versie 3 resultaten toe"""
    output_lines = []
    
    with open('dataset.txt', 'r') as f:
        # Lees header
        header = f.readline().strip()
        output_lines.append(header + "\tMatchV3\tConfidence")
        
        # Verwerk elke regel
        for line in f:
            if not line.strip():
                continue
                
            parts = line.strip().split('\t')
            if len(parts) < 4:
                print(f"Ongeldige regel overgeslagen: {line.strip()}")
                continue
                
            url1, url2, match_real, match_v1 = parts
            
            try:
                # Download beide afbeeldingen (of gebruik lokale kopie)
                img1 = download_image(url1)
                img2 = download_image(url2)
                
                # Vergelijk met versie 3
                match_v3, confidence = compare_images_v3(img1, img2)
                
                # Voeg resultaat toe aan de regel
                new_line = f"{line.strip()}\t{1 if match_v3 else 0}\t{confidence}"
                output_lines.append(new_line)
                
                print(f"Verwerkt: {url1} vs {url2} - MatchV3: {match_v3} (confidence: {confidence}%)")
                
            except Exception as e:
                print(f"Fout bij verwerken van {url1} vs {url2}: {str(e)}")
                # Voeg een foutmelding toe als kolom
                new_line = f"{line.strip()}\tERROR\t0"
                output_lines.append(new_line)
    
    # Schrijf resultaten naar nieuw bestand
    with open('dataset_v3.txt', 'w') as f:
        f.write('\n'.join(output_lines))
    
    print("\nVerwerking voltooid! Resultaten opgeslagen in dataset_v3.txt")

if __name__ == '__main__':
    process_dataset() 