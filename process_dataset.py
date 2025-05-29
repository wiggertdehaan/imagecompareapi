import cv2
import requests
import os
import logging
from app import compare_images_v3

# Configureer logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_unique_image_path(url):
    """Genereer een uniek pad voor een afbeelding op basis van de URL"""
    # Haal de bestandsnaam uit de URL
    filename = url.split('/')[-1]
    # Verwijder query parameters als die er zijn
    filename = filename.split('?')[0]
    # Maak een unieke naam door de URL te hashen
    import hashlib
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    # Combineer hash met originele bestandsnaam
    name, ext = os.path.splitext(filename)
    return f"images/{url_hash}_{name}{ext}"

def download_image(url, save_path):
    """Download een afbeelding en sla deze op"""
    if os.path.exists(save_path):
        logger.info(f"Gebruik lokale kopie: {save_path}")
        return cv2.imread(save_path)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Maak images directory als die nog niet bestaat
        os.makedirs('images', exist_ok=True)
        
        # Sla de afbeelding op
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Afbeelding gedownload en opgeslagen: {save_path}")
        return cv2.imread(save_path)
    except Exception as e:
        logger.error(f"Fout bij downloaden van {url}: {str(e)}")
        return None

def process_dataset(input_file='dataset.txt', output_file='dataset_v3.txt'):
    """Verwerk het dataset bestand en vergelijk de afbeeldingen"""
    if not os.path.exists(input_file):
        logger.error(f"Input bestand {input_file} niet gevonden!")
        return

    results = []
    with open(input_file, 'r') as f:
        # Lees de eerste regel om het formaat te bepalen
        first_line = f.readline().strip()
        f.seek(0)  # Ga terug naar het begin van het bestand
        
        # Bepaal het scheidingsteken (tab of komma)
        if '\t' in first_line:
            delimiter = '\t'
            logger.info("Gebruik tab-gescheiden formaat")
        else:
            delimiter = ','
            logger.info("Gebruik komma-gescheiden formaat")
        
        # Lees de header
        header = f.readline().strip()
        header_parts = header.split(delimiter)
        
        # Voeg de nieuwe kolommen toe aan de header
        new_header = delimiter.join(header_parts + ['MatchV3', 'ConfidenceV3'])
        results.append(new_header)
        
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            try:
                # Splits de regel op basis van het gedetecteerde scheidingsteken
                parts = line.split(delimiter)
                
                # Controleer of we genoeg kolommen hebben
                if len(parts) < 2:
                    logger.warning(f"Ongeldige regel overgeslagen (te weinig kolommen): {line}")
                    continue
                
                # Haal de URLs uit de eerste twee kolommen
                url1 = parts[0].strip()
                url2 = parts[1].strip()
                
                # Download of laad de afbeeldingen
                path1 = get_unique_image_path(url1)
                path2 = get_unique_image_path(url2)
                
                img1 = download_image(url1, path1)
                img2 = download_image(url2, path2)
                
                if img1 is None or img2 is None:
                    logger.error(f"Kon een van de afbeeldingen niet laden: {url1} of {url2}")
                    continue
                
                # Vergelijk de afbeeldingen
                match_v3, confidence = compare_images_v3(img1, img2)
                
                # Behoud alle originele kolommen en voeg de nieuwe resultaten toe
                new_line = delimiter.join(parts + [str(int(match_v3)), str(confidence)])
                results.append(new_line)
                logger.info(f"Resultaat: {new_line}")
                
            except Exception as e:
                logger.error(f"Fout bij verwerken van regel: {line}")
                logger.error(str(e))
                continue
    
    # Schrijf alle resultaten naar het output bestand
    with open(output_file, 'w') as f:
        for result in results:
            f.write(result + "\n")
    
    logger.info(f"Verwerking voltooid. Resultaten opgeslagen in {output_file}")

if __name__ == "__main__":
    process_dataset() 