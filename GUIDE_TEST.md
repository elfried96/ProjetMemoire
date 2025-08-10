# Guide de Test - Surveillance Orchestrator

## Problème résolu ✅
L'erreur `AttributeError: 'Settings' object has no attribute 'ModelType'` a été corrigée dans `kim_wrapper.py` en important directement `ModelType`.

## Étapes pour tester le système

### 1. Installation des dépendances minimales
```bash
pip install opencv-python pillow numpy
```

### 2. Installation des dépendances IA (optionnel mais recommandé)
```bash
pip install torch torchvision transformers accelerate
```

### 3. Test basique (sans IA)
```bash
python test_basic.py
```

### 4. Test avec modèles IA
```bash
# Liste des modèles disponibles
python main.py --list-models

# Test avec une vidéo
python main.py videos/surveillance_test.mp4 --section "Rayon cosmétique"
```

## Tests disponibles

### Test simple de configuration
```bash
python3 -c "from src.config.settings import settings; print(f'VLM principal: {settings.config.primary_vlm.value}')"
```

### Test du système de logging
```bash
python3 -c "from src.utils.logging import get_surveillance_logger; logger = get_surveillance_logger(); logger.info('Test OK')"
```

### Test avec une vraie vidéo (nécessite OpenCV + modèles IA)
```bash
python main.py "videos/surveillance_test.mp4" --verbose
```

## Structure des tests

1. **test_basic.py** - Test sans modèles IA (configuration, logging, preprocessing)
2. **main.py** - Application complète avec IA
3. **demo.py** - Démonstration interactive (si présent)

## Erreurs communes et solutions

### `No module named 'cv2'`
```bash
pip install opencv-python
```

### `No module named 'torch'`
```bash
pip install torch torchvision
```

### `No module named 'transformers'`
```bash
pip install transformers accelerate
```

### Mémoire GPU insuffisante
- Utilisez SmolVLM au lieu de KIM
- Réduisez la taille des batches
- Activez le nettoyage automatique

## Utilisation recommandée pour premier test

1. Installez OpenCV : `pip install opencv-python pillow`
2. Lancez : `python test_basic.py`
3. Si succès, installez l'IA : `pip install torch transformers`
4. Testez : `python main.py videos/surveillance_test.mp4 --verbose`