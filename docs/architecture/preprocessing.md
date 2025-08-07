# Architecture du Preprocessing

## Vue d'ensemble

Le module de preprocessing (`src/utils/preprocessing.py`) gère l'extraction et l'optimisation des frames vidéo pour l'analyse par les modèles VLM.

## Composants Principaux

### 1. Extracteur de Frames

```python
def extract_frames_from_video(video_path: str, seconds_per_frame: float = 2.0, max_frames: int = 10) -> List[Image.Image]:
    """
    Extrait des frames d'une vidéo à intervalles réguliers
    """
```

**Fonctionnalités** :
- Extraction à intervalle fixe ou intelligent
- Limite automatique du nombre de frames
- Gestion des formats vidéo courants
- Optimisation mémoire avec PIL

### 2. Détection de Keyframes

```python  
def extract_keyframes(video_path: str, target_frames: int = 5) -> List[Image.Image]:
    """
    Sélection intelligente des frames les plus représentatives
    """
```

**Algorithme** :
1. **Analyse de variance** : Détection des changements significatifs
2. **Histogramme** : Comparaison des distributions de couleurs  
3. **Seuil adaptatif** : Ajustement automatique selon le contenu
4. **Optimisation temporelle** : Répartition équilibrée dans la vidéo

### 3. Optimisation des Images

```python
def preprocess_frame(frame: Image.Image, target_size: Tuple[int, int] = (384, 384)) -> Image.Image:
    """
    Optimise une frame pour l'analyse VLM
    """
```

**Traitements appliqués** :
- **Redimensionnement** : Taille optimale pour les modèles
- **Normalisation** : Ajustement automatique luminosité/contraste
- **Format** : Conversion RGB si nécessaire
- **Compression** : Équilibre qualité/performance

## Pipeline Complet

```mermaid
graph LR
    A[Vidéo] --> B[Validation Format]
    B --> C[Extraction Frames]
    C --> D[Détection Keyframes]
    D --> E[Preprocessing]
    E --> F[Optimisation Mémoire]
    F --> G[Frames Optimisées]
```

### Étapes Détaillées

#### 1. Validation d'Entrée
```python
def validate_video_file(video_path: str) -> bool:
    """Vérifie format, taille et intégrité du fichier"""
    if not video_path.exists():
        raise FileNotFoundError(f"Vidéo non trouvée: {video_path}")
    
    # Vérification du format
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    if not any(video_path.suffix.lower() == ext for ext in valid_extensions):
        raise ValueError(f"Format non supporté: {video_path.suffix}")
```

#### 2. Extraction Intelligente
```python  
def intelligent_frame_extraction(video_path: str, config: ProcessingConfig) -> List[Image.Image]:
    """
    Extraction adaptative selon les ressources disponibles
    """
    # Auto-configuration selon GPU
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory < 4:
            config.max_frames = min(config.max_frames, 5)
            config.target_size = (256, 256)
```

#### 3. Optimisation Performance
```python
def batch_preprocess_frames(frames: List[Image.Image], batch_size: int = 4) -> List[Image.Image]:
    """
    Traitement par lots pour optimiser les performances
    """
    processed_frames = []
    
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        # Traitement parallèle du lot
        with concurrent.futures.ThreadPoolExecutor() as executor:
            batch_processed = list(executor.map(preprocess_frame, batch))
        processed_frames.extend(batch_processed)
```

## Gestion Mémoire

### Monitoring en Temps Réel
```python
@memory_monitor("Preprocessing")
def extract_and_preprocess(video_path: str) -> List[Image.Image]:
    """
    Extraction avec surveillance mémoire automatique
    """
    try:
        frames = extract_frames_from_video(video_path)
        return [preprocess_frame(frame) for frame in frames]
    finally:
        # Nettoyage automatique
        gc.collect()
```

### Optimisations Automatiques
- **Réduction qualité** : Si pression mémoire détectée
- **Batch adaptatif** : Ajustement taille selon ressources
- **Cache intelligent** : Réutilisation frames similaires
- **Cleanup proactif** : Libération mémoire immédiate

## Configuration et Personnalisation

### Paramètres Principaux
```python
@dataclass
class ProcessingConfig:
    seconds_per_frame: float = 2.0        # Intervalle extraction
    max_frames: int = 10                  # Limite frames
    target_size: Tuple[int, int] = (384, 384)  # Taille optimale
    use_keyframes: bool = False           # Sélection intelligente
    quality_threshold: float = 0.8        # Seuil qualité
    enable_enhancement: bool = True       # Améliorations automatiques
```

### Adaptation Contextuelle
```python
def adapt_config_for_context(section: str, time_of_day: str) -> ProcessingConfig:
    """
    Configuration adaptée au contexte d'analyse
    """
    config = ProcessingConfig()
    
    # Zones critiques = plus de frames
    if "caisse" in section.lower() or "sortie" in section.lower():
        config.max_frames = 15
        config.seconds_per_frame = 1.5
    
    # Nuit = amélioration qualité
    if time_of_day.lower() in ["nuit", "soirée"]:
        config.enable_enhancement = True
        config.quality_threshold = 0.9
```

## Performance et Monitoring

### Métriques Automatiques
- **Temps extraction** : Durée par vidéo
- **Qualité frames** : Score moyen des frames extraites  
- **Utilisation mémoire** : Peak et moyenne
- **Taux succès** : Pourcentage extractions réussies

### Logging Intelligent
```python
logger.info(f"📽️  Extraction: {len(frames)} frames en {duration:.2f}s")
logger.debug(f"🎯 Keyframes: sélection {keyframes_count}/{total_frames}")
logger.warning(f"⚠️  Mémoire: {memory_usage:.1f}MB (seuil: {threshold:.1f}MB)")
```