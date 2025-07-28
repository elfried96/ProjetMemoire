import sys
sys.path.append("src")

import logging
from tqdm import tqdm
from models.kim_wrapper import KIMWrapper

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

kim = KIMWrapper()
video_path = "videos/surveillance_test.mp4"
section = "Rayon cosmétiques"
time_of_day = "Fin d’après-midi"
crowd_density = "dense"

logging.info("Début de l'analyse vidéo : %s", video_path)

# Extraction des frames avec barre de progression
frames = kim.extract_frames(video_path)
logging.info("%d frames extraites et prétraitées.", len(frames))

# Barre de progression pour l'analyse des frames
results = []
for i in tqdm(range(len(frames)), desc="Analyse des frames", unit="frame"):
    # Ici, on pourrait analyser chaque frame individuellement si besoin
    pass  # Le traitement global est fait dans analyze_video, donc on ne fait rien ici

result = kim.analyze_video(
    video_path=video_path,
    section=section,
    time_of_day=time_of_day,
    crowd_density=crowd_density
)

if result:
    logging.info("Analyse terminée. Affichage des résultats.")
    print("\n-------- PENSÉE DU MODÈLE --------")
    print(result["thinking"])
    print("\n-------- ANALYSE GÉNÉRALE --------")
    print(result["summary"])
