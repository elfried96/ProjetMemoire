#!/usr/bin/env python3
"""
Test de base du système de surveillance (sans modèles IA).
Teste le préprocessing vidéo et les composants principaux.
"""

import sys
from pathlib import Path

# Test des imports
try:
    from src.config.settings import settings
    from src.utils.logging import get_surveillance_logger
    from src.utils.preprocessing import video_processor
    print("✅ Tous les imports réussis")
except Exception as e:
    print(f"❌ Erreur d'import: {e}")
    sys.exit(1)

def test_video_info():
    """Test d'analyse des informations vidéo."""
    logger = get_surveillance_logger()
    
    video_path = Path("videos/surveillance_test.mp4")
    
    if not video_path.exists():
        print(f"❌ Vidéo de test non trouvée: {video_path}")
        return False
    
    try:
        # Test des informations vidéo
        info = video_processor.get_video_info(video_path)
        print(f"✅ Vidéo analysée: {info['duration']:.1f}s, {info['fps']:.1f}fps")
        print(f"   Résolution: {info['resolution']}")
        print(f"   Taille: {info['size_mb']:.1f} MB")
        return True
    except Exception as e:
        print(f"❌ Erreur analyse vidéo: {e}")
        return False

def test_configuration():
    """Test de la configuration."""
    try:
        print(f"✅ VLM principal: {settings.config.primary_vlm.value}")
        print(f"✅ Batch size: {settings.config.batch_size}")
        print(f"✅ Traitement: {settings.config.processing.seconds_per_frame}s/frame")
        return True
    except Exception as e:
        print(f"❌ Erreur configuration: {e}")
        return False

def main():
    """Test principal."""
    print("🎬 Test de base du Surveillance Orchestrator")
    print("=" * 50)
    
    # Test de configuration
    if not test_configuration():
        return 1
    
    # Test d'analyse vidéo
    if not test_video_info():
        return 1
        
    print("\n✅ Tous les tests de base réussis!")
    print("💡 Pour tester avec l'IA, installez les dépendances:")
    print("   pip install torch torchvision transformers opencv-python")
    
    return 0

if __name__ == "__main__":
    exit(main())