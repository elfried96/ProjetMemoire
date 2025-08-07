# Configuration

## Configuration de Base

### Variables d'Environnement

Créez un fichier `.env` à la racine du projet :

```bash
# Modèle VLM principal
SURVEILLANCE_PRIMARY_VLM=smolvlm

# Niveau de log (DEBUG, INFO, WARNING, ERROR)
SURVEILLANCE_LOG_LEVEL=INFO

# Taille de lot pour le traitement
SURVEILLANCE_BATCH_SIZE=4

# Fraction de VRAM utilisable (0.0-1.0)
SURVEILLANCE_MAX_GPU_MEMORY=0.8
```

### Configuration Programmatique

```python
from src.config import settings, ModelType

# Configuration des modèles
settings.config.primary_vlm = ModelType.SMOLVLM
settings.config.batch_size = 4
settings.config.cleanup_after_analysis = True

# Configuration du preprocessing
settings.config.processing.seconds_per_frame = 2.0
settings.config.processing.max_frames = 10
settings.config.processing.target_size = (384, 384)
```

## Optimisations selon les Ressources

### GPU Limité (< 4GB)
```python
settings.config.batch_size = 1
settings.config.cleanup_after_analysis = True
settings.config.processing.max_frames = 5
```

### GPU Moyen (4-8GB)
```python
settings.config.batch_size = 2
settings.config.cleanup_after_analysis = True
settings.config.processing.max_frames = 8
```

### GPU Puissant (> 8GB)
```python
settings.config.batch_size = 6
settings.config.cleanup_after_analysis = False
settings.config.processing.max_frames = 15
```

## Configuration de la Mémoire Contextuelle

### Paramètres d'Apprentissage
```python
# Dans memory_engine.py
pattern_detector.learning_threshold = 5  # Minimum d'occurrences
contextual_learning.cache_duration = timedelta(minutes=5)
```

### Optimisation de la Mémoire
```python
# Limite les événements gardés en mémoire
memory_manager.max_events = 1000
memory_manager.max_patterns = 100
memory_manager.alert_retention_days = 30
```