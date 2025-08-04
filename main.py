# main.py

import sys
import logging
from pathlib import Path

from src.orchestrator.controller import SurveillanceOrchestrator

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

def main():
    if len(sys.argv) < 2:
        print("⚠️ Usage: python main.py <chemin_vers_video>")
        sys.exit(1)

    video_path = sys.argv[1]
    if not Path(video_path).exists():
        print(f"❌ La vidéo '{video_path}' n'existe pas.")
        sys.exit(1)

    # Informations contextuelles (peuvent être remplacées par arguments plus tard)
    section = "Rayon cosmétique"
    time_of_day = "Fin d’après-midi"
    crowd_density = "dense"

    # Instanciation de l'orchestrateur
    orchestrator = SurveillanceOrchestrator()

    # Analyse intelligente
    result = orchestrator.analyze(
        video_path=video_path,
        section=section,
        time_of_day=time_of_day,
        crowd_density=crowd_density
    )

    # Résultat final
    if result:
        print("\n🧠 PENSÉE DU VLM :\n", result["thinking"])
        print("\n📊 DESCRIPTION DE LA SCÈNE :\n", result["summary"])
        print("\n🧠 DÉCISION DE L'ORCHESTRATEUR (LLM):\n", result["decision"])
    else:
        print("❌ Échec de l’analyse.")

if __name__ == "__main__":
    main()
