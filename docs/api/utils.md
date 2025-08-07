# API Utilitaires

## Vue d'ensemble

Les modules utilitaires fournissent des fonctionnalités de support pour le preprocessing, l'optimisation mémoire, et le monitoring du système.

## Module `src.utils.preprocessing`

Utilitaires pour l'extraction et le traitement des frames vidéo.

### Fonctions Principales

#### `extract_frames_from_video()`

Extrait des frames d'une vidéo à intervalles réguliers.

```python
def extract_frames_from_video(
    video_path: str,
    seconds_per_frame: float = 2.0,
    max_frames: int = 10
) -> List[Image.Image]:
```

**Paramètres** :
- `video_path` : Chemin vers le fichier vidéo
- `seconds_per_frame` : Intervalle d'extraction en secondes
- `max_frames` : Limite du nombre de frames extraites

**Retourne** : Liste d'images PIL

**Exemple** :
```python
from src.utils.preprocessing import extract_frames_from_video

frames = extract_frames_from_video(
    "surveillance.mp4",
    seconds_per_frame=1.5,
    max_frames=8
)
print(f"Extracted {len(frames)} frames")
```

#### `extract_keyframes()`

Sélection intelligente des frames les plus représentatives.

```python
def extract_keyframes(
    video_path: str,
    target_frames: int = 5
) -> List[Image.Image]:
```

**Algorithme** :
- Analyse de variance entre frames consécutives
- Détection des changements significatifs de contenu
- Répartition équilibrée dans la durée de la vidéo

**Exemple** :
```python
keyframes = extract_keyframes("long_video.mp4", target_frames=3)
# Sélectionne les 3 frames les plus représentatives
```

#### `preprocess_frame()`

Optimise une frame pour l'analyse par les modèles VLM.

```python
def preprocess_frame(
    frame: Image.Image,
    target_size: Tuple[int, int] = (384, 384)
) -> Image.Image:
```

**Traitements appliqués** :
- Redimensionnement à la taille cible
- Conversion en RGB si nécessaire
- Normalisation de l'éclairage
- Ajustement du contraste

#### `batch_preprocess_frames()`

Traitement par lots pour optimiser les performances.

```python
def batch_preprocess_frames(
    frames: List[Image.Image],
    batch_size: int = 4
) -> List[Image.Image]:
```

**Avantages** :
- Traitement parallèle des frames
- Optimisation mémoire
- Monitoring automatique des ressources

## Module `src.utils.memory_optimizer`

Gestionnaire intelligent de la mémoire GPU et CPU.

### Classe `MemoryOptimizer`

```python
from src.utils.memory_optimizer import memory_optimizer

# Instance globale disponible
optimizer = memory_optimizer
```

### Méthodes Principales

#### `check_memory_pressure()`

Vérifie si le système est sous pression mémoire.

```python
def check_memory_pressure(self) -> bool:
```

**Retourne** : `True` si la mémoire est saturée

**Exemple** :
```python
if memory_optimizer.check_memory_pressure():
    print("⚠️ Mémoire saturée, optimisation nécessaire")
    memory_optimizer.aggressive_cleanup()
```

#### `aggressive_cleanup()`

Lance un nettoyage agressif de la mémoire.

```python
def aggressive_cleanup(self) -> Dict[str, float]:
```

**Actions** :
- Collecte du garbage collector Python
- Nettoyage cache PyTorch
- Libération mémoire GPU
- Déchargement modèles non utilisés

**Retourne** : Statistiques de nettoyage

#### `auto_configure_settings()`

Configuration automatique selon les ressources disponibles.

```python
def auto_configure_settings(self) -> None:
```

**Ajustements automatiques** :
- Taille des lots de traitement
- Activation du nettoyage automatique
- Fraction de mémoire GPU utilisable
- Nombre maximum de frames

#### `get_memory_stats()`

Statistiques détaillées de l'utilisation mémoire.

```python
def get_memory_stats(self) -> Dict[str, Any]:
```

**Informations retournées** :
- Mémoire CPU utilisée/disponible
- Mémoire GPU utilisée/disponible (si CUDA)
- Nombre d'objets en mémoire Python
- État des caches PyTorch

### Décorateur `@memory_monitor`

Monitoring automatique de l'utilisation mémoire.

```python
from src.utils.memory_optimizer import memory_monitor

@memory_monitor("Analyse VLM")
def analyze_with_vlm(frames):
    # Code d'analyse
    return vlm.analyze(frames)

# Ou en context manager
with memory_monitor("Preprocessing"):
    frames = extract_frames_from_video(video_path)
```

**Fonctionnalités** :
- Monitoring avant/après exécution
- Logs automatiques des variations mémoire
- Alerte en cas de fuite mémoire
- Statistiques de performance

## Utilitaires de Configuration GPU

### `check_gpu_availability()`

Vérifie la disponibilité et les capacités GPU.

```python
def check_gpu_availability() -> Dict[str, Any]:
```

**Informations retournées** :
```python
{
    "cuda_available": True,
    "device_count": 1,
    "current_device": 0,
    "device_name": "NVIDIA GeForce RTX 3080",
    "total_memory_gb": 10.0,
    "free_memory_gb": 8.2,
    "compute_capability": (8, 6)
}
```

### `get_optimal_device()`

Sélectionne automatiquement le meilleur device disponible.

```python
def get_optimal_device() -> str:
```

**Logique de sélection** :
- GPU CUDA si disponible et suffisant VRAM
- GPU MPS (Apple Silicon) si disponible
- CPU en fallback

## Module Logging

### Configuration Intelligente

```python
from src.utils.logging import setup_logging

# Configuration automatique selon l'environnement
setup_logging(level="INFO", enable_memory_logs=True)
```

### Formatters Personnalisés

Le système utilise des formatters enrichis :

```python
# Exemple de log avec contexte
logger.info("📽️ Extraction: 8 frames en 2.34s")
logger.warning("⚠️ Mémoire GPU: 85% utilisée")
logger.error("❌ Échec chargement modèle KIM: VRAM insuffisante")
```

## Exemples d'Usage Combinés

### Pipeline Complet Optimisé

```python
from src.utils.preprocessing import extract_keyframes, batch_preprocess_frames
from src.utils.memory_optimizer import memory_monitor

@memory_monitor("Pipeline complet")
def optimized_video_processing(video_path: str):
    # 1. Extraction intelligente
    frames = extract_keyframes(video_path, target_frames=5)
    
    # 2. Preprocessing par lots
    processed_frames = batch_preprocess_frames(frames, batch_size=2)
    
    # 3. Vérification mémoire avant analyse
    if memory_optimizer.check_memory_pressure():
        memory_optimizer.aggressive_cleanup()
    
    return processed_frames
```

### Monitoring de Session

```python
def monitor_analysis_session():
    """Monitoring complet d'une session d'analyse"""
    
    initial_stats = memory_optimizer.get_memory_stats()
    print(f"Mémoire initiale: {initial_stats['cpu_used_gb']:.1f}GB CPU")
    
    if initial_stats['gpu_available']:
        print(f"GPU: {initial_stats['gpu_used_gb']:.1f}/{initial_stats['gpu_total_gb']:.1f}GB")
    
    # Session d'analyse
    for video in video_list:
        with memory_monitor(f"Analyse {video}"):
            result = orchestrator.analyze(video, section, time, density)
            
            # Nettoyage périodique
            if memory_optimizer.check_memory_pressure():
                stats = memory_optimizer.aggressive_cleanup()
                print(f"🧹 Nettoyage: {stats['memory_freed_mb']:.1f}MB libérés")
    
    final_stats = memory_optimizer.get_memory_stats()
    memory_diff = final_stats['cpu_used_gb'] - initial_stats['cpu_used_gb']
    print(f"Variation mémoire session: {memory_diff:+.1f}GB")
```

### Configuration Adaptative

```python
def adaptive_configuration():
    """Configuration qui s'adapte aux ressources disponibles"""
    
    gpu_info = check_gpu_availability()
    
    if gpu_info['cuda_available']:
        vram_gb = gpu_info['free_memory_gb']
        
        if vram_gb >= 8:
            # Configuration haute performance
            settings.config.batch_size = 6
            settings.config.primary_vlm = ModelType.KIM
            settings.config.cleanup_after_analysis = False
            
        elif vram_gb >= 4:
            # Configuration équilibrée
            settings.config.batch_size = 3
            settings.config.primary_vlm = ModelType.SMOLVLM
            settings.config.cleanup_after_analysis = True
            
        else:
            # Configuration économique
            settings.config.batch_size = 1
            settings.config.primary_vlm = ModelType.SMOLVLM
            settings.config.cleanup_after_analysis = True
            
    else:
        # Configuration CPU uniquement
        settings.config.batch_size = 1
        settings.config.cleanup_after_analysis = True
        
    print(f"⚙️ Configuration adaptée: batch_size={settings.config.batch_size}")
```