# API Configuration

## Module `src.config.settings`

Configuration centralisée du système avec dataclasses typées.

### Classes Principales

#### `SurveillanceConfig`

Configuration principale du système.

```python
@dataclass
class SurveillanceConfig:
    primary_vlm: ModelType = ModelType.SMOLVLM
    fallback_vlm: ModelType = ModelType.KIM
    llm_model: str = "microsoft/phi-2"
    processing: ProcessingConfig = None
    batch_size: int = 4
    cleanup_after_analysis: bool = True
    max_gpu_memory_fraction: float = 0.8
    log_level: str = "INFO"
```

**Attributs** :
- `primary_vlm` : Modèle VLM principal (SmolVLM par défaut)
- `fallback_vlm` : Modèle de secours si primaire indisponible  
- `llm_model` : Modèle Hugging Face pour les décisions
- `processing` : Configuration du preprocessing
- `batch_size` : Taille des lots de traitement
- `cleanup_after_analysis` : Nettoyage automatique post-analyse
- `max_gpu_memory_fraction` : Fraction maximale VRAM utilisable
- `log_level` : Niveau de logging

#### `ProcessingConfig`

Configuration du preprocessing vidéo.

```python
@dataclass  
class ProcessingConfig:
    seconds_per_frame: float = 2.0
    max_frames: int = 10
    target_size: Tuple[int, int] = (384, 384)
    use_keyframes: bool = False
    quality_threshold: float = 0.8
    enable_enhancement: bool = True
```

**Attributs** :
- `seconds_per_frame` : Intervalle d'extraction en secondes
- `max_frames` : Nombre maximum de frames extraites
- `target_size` : Taille cible des images (largeur, hauteur)
- `use_keyframes` : Utiliser la sélection intelligente de frames
- `quality_threshold` : Seuil de qualité minimum (0.0-1.0)
- `enable_enhancement` : Activer les améliorations automatiques

### Énumérations

#### `ModelType`

Types de modèles VLM disponibles.

```python
class ModelType(Enum):
    SMOLVLM = "smolvlm"
    KIM = "kim"
```

### Singleton Global

#### `settings`

Instance globale de configuration accessible depuis toute l'application.

```python
from src.config import settings

# Accès configuration
config = settings.config
print(f"Modèle principal: {config.primary_vlm}")

# Modification runtime
settings.config.batch_size = 2
settings.config.log_level = "DEBUG"
```

### Variables d'Environnement

Le système lit automatiquement les variables d'environnement :

```bash
# Fichier .env
SURVEILLANCE_PRIMARY_VLM=smolvlm
SURVEILLANCE_LOG_LEVEL=INFO  
SURVEILLANCE_BATCH_SIZE=4
SURVEILLANCE_MAX_GPU_MEMORY=0.8
SURVEILLANCE_CLEANUP_AFTER_ANALYSIS=true
```

**Variables supportées** :
- `SURVEILLANCE_PRIMARY_VLM` : Modèle VLM principal
- `SURVEILLANCE_LOG_LEVEL` : Niveau de logging
- `SURVEILLANCE_BATCH_SIZE` : Taille des lots
- `SURVEILLANCE_MAX_GPU_MEMORY` : Fraction VRAM max
- `SURVEILLANCE_CLEANUP_AFTER_ANALYSIS` : Nettoyage auto

### Méthodes Utilitaires

#### `load_from_env()`

Charge la configuration depuis les variables d'environnement.

```python
settings.load_from_env()
```

#### `validate_config()`

Valide la cohérence de la configuration.

```python  
if not settings.validate_config():
    raise ValueError("Configuration invalide")
```

### Exemples d'Usage

#### Configuration Programmatique

```python
from src.config import settings, ModelType, ProcessingConfig

# Configuration des modèles
settings.config.primary_vlm = ModelType.SMOLVLM
settings.config.batch_size = 2

# Configuration du preprocessing  
settings.config.processing = ProcessingConfig(
    seconds_per_frame=1.5,
    max_frames=8,
    use_keyframes=True
)
```

#### Configuration par Contexte

```python
def configure_for_limited_resources():
    """Configuration optimisée ressources limitées"""
    settings.config.batch_size = 1
    settings.config.cleanup_after_analysis = True
    settings.config.max_gpu_memory_fraction = 0.6
    settings.config.processing.max_frames = 5

def configure_for_high_performance():
    """Configuration haute performance"""
    settings.config.batch_size = 8
    settings.config.cleanup_after_analysis = False
    settings.config.max_gpu_memory_fraction = 0.9
    settings.config.processing.max_frames = 20
```

#### Validation et Debug

```python
from src.config import settings

# Affichage configuration actuelle
print(f"Configuration actuelle:")
print(f"  VLM principal: {settings.config.primary_vlm}")
print(f"  Batch size: {settings.config.batch_size}")
print(f"  Nettoyage auto: {settings.config.cleanup_after_analysis}")

# Validation
if settings.config.batch_size > 8:
    print("⚠️ Batch size élevé, surveillance mémoire recommandée")

if settings.config.max_gpu_memory_fraction > 0.9:
    print("⚠️ Usage GPU élevé, risque de manque mémoire")
```