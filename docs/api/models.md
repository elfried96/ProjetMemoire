# API Modèles

## Vue d'ensemble

L'API des modèles fournit une interface unifiée pour interagir avec les modèles VLM (Vision-Language) et LLM utilisés dans le système.

## Classes de Base

### `BaseVLMModel`

Interface abstraite pour tous les modèles VLM.

```python
from abc import ABC, abstractmethod
from src.models.base import BaseVLMModel, AnalysisResult

class BaseVLMModel(ABC):
    @abstractmethod
    def analyze_images(self, images: List[Image.Image], prompt: str) -> AnalysisResult:
        """Analyse des images avec un prompt donné"""
        pass
    
    @abstractmethod
    def load_model(self) -> None:
        """Charge le modèle en mémoire"""
        pass
    
    @abstractmethod  
    def unload_model(self) -> None:
        """Décharge le modèle de la mémoire"""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Vérifie si le modèle est chargé"""
        pass
```

### `BaseLLMModel`

Interface abstraite pour les modèles LLM de décision.

```python
from src.models.base import BaseLLMModel, SuspicionAnalysis

class BaseLLMModel(ABC):
    @abstractmethod
    def analyze_context(self, context: Dict) -> SuspicionAnalysis:
        """Analyse contextuelle et prise de décision"""
        pass
    
    @abstractmethod
    def load_model(self) -> None:
        """Charge le modèle LLM"""
        pass
```

## Modèles VLM

### `SmolVLMWrapper`

Wrapper pour le modèle SmolVLM optimisé pour les ressources limitées.

```python
from src.models.smolvlm_wrapper import SmolVLMWrapper

# Initialisation
vlm = SmolVLMWrapper(
    model_name="HuggingFaceTB/SmolVLM-Instruct",
    device="auto"
)

# Analyse d'images
result = vlm.analyze_surveillance_scene(
    images=[frame1, frame2],
    section="Caisse principale",
    time_of_day="Après-midi",
    crowd_density="dense"
)
```

**Méthodes principales** :

#### `analyze_surveillance_scene()`
```python
def analyze_surveillance_scene(
    self, 
    images: List[Image.Image], 
    section: str, 
    time_of_day: str, 
    crowd_density: str
) -> AnalysisResult:
```

**Paramètres** :
- `images` : Liste des frames à analyser
- `section` : Zone de surveillance (ex: "Caisse", "Entrée")
- `time_of_day` : Moment de la journée (ex: "Matin", "Soirée")  
- `crowd_density` : Densité de foule (ex: "faible", "dense")

**Retourne** : `AnalysisResult` avec description et métadonnées

#### `is_available()`
```python
@staticmethod
def is_available() -> bool:
    """Vérifie si SmolVLM peut être utilisé"""
```

### `KIMWrapper`

Wrapper pour le modèle KIM haute performance.

```python
from src.models.kim_wrapper import KIMWrapper

# Vérification disponibilité
if KIMWrapper.is_available():
    vlm = KIMWrapper()
    result = vlm.analyze_surveillance_scene(images, "Sortie", "Soir", "moyenne")
```

**Spécificités** :
- Nécessite plus de VRAM (~6-8GB)
- Meilleure précision pour analyses complexes
- Chargement automatique différé
- Vérification ressources GPU

## Modèles LLM

### `LLMDecisionModel`

Modèle de prise de décision basé sur Phi-3.

```python
from src.models.llm_decision_model import LLMDecisionModel

# Initialisation
llm = LLMDecisionModel(model_name="microsoft/phi-2")

# Analyse de contexte
analysis = llm.analyze_context({
    "vlm_description": "Personne manipulant des objets près de la sortie",
    "section": "Sortie principale", 
    "time_of_day": "Soirée",
    "crowd_density": "faible",
    "historical_context": {...}
})
```

#### `analyze_context()`
```python
def analyze_context(self, context: Dict) -> SuspicionAnalysis:
```

**Paramètres contexte** :
- `vlm_description` : Description VLM de la scène
- `section` : Zone analysée
- `time_of_day` : Moment de la journée
- `crowd_density` : Densité de foule
- `historical_context` : Données historiques de la zone

**Retourne** : `SuspicionAnalysis` avec niveau de suspicion et recommandations

## Factory Pattern

### `create_vlm_model()`

Factory pour créer les modèles VLM selon la configuration.

```python
from src.models import create_vlm_model
from src.config import ModelType

# Création automatique selon config
vlm = create_vlm_model()

# Création avec modèle spécifique  
vlm = create_vlm_model(ModelType.KIM)

# Création avec paramètres
vlm = create_vlm_model(
    model_type=ModelType.SMOLVLM,
    device="cuda:0"
)
```

## Structures de Données

### `AnalysisResult`

Résultat d'analyse VLM.

```python
@dataclass
class AnalysisResult:
    description: str                    # Description de la scène
    confidence: float                   # Niveau de confiance (0.0-1.0)
    detected_objects: List[str]         # Objets détectés
    people_count: int                   # Nombre de personnes
    suspicious_activities: List[str]    # Activités suspectes
    metadata: Dict[str, Any]           # Métadonnées additionnelles
    processing_time: float             # Temps de traitement
```

### `SuspicionAnalysis`

Analyse de suspicion par le LLM.

```python
@dataclass  
class SuspicionAnalysis:
    suspicion_level: str               # "low", "medium", "high" 
    confidence: float                  # Confiance dans l'analyse
    alert_type: str                   # Type d'alerte
    recommendation: str               # Recommandation d'action
    reasoning: str                    # Justification de la décision
    priority_score: int               # Score de priorité (1-10)
    contextual_factors: List[str]     # Facteurs contextuels considérés
```

## Utilitaires de Gestion

### Monitoring des Ressources

```python
from src.models.utils import check_gpu_memory, optimize_model_loading

# Vérification mémoire GPU
gpu_info = check_gpu_memory()
print(f"VRAM libre: {gpu_info['free_memory']:.1f}GB")

# Optimisation chargement
optimize_model_loading(target_memory_gb=4.0)
```

### Switching Automatique

```python
from src.models.utils import get_optimal_vlm_model

# Sélection automatique du meilleur modèle
optimal_model = get_optimal_vlm_model()
vlm = create_vlm_model(optimal_model)
```

## Exemples d'Usage

### Analyse Complète

```python
from src.models import create_vlm_model, create_llm_model
from src.utils.preprocessing import extract_frames_from_video

# Création des modèles
vlm = create_vlm_model()
llm = create_llm_model()

# Extraction frames
frames = extract_frames_from_video("surveillance.mp4")

# Analyse VLM
vlm_result = vlm.analyze_surveillance_scene(
    frames, "Caisse 1", "Après-midi", "dense"
)

# Décision LLM
decision = llm.analyze_context({
    "vlm_description": vlm_result.description,
    "section": "Caisse 1",
    "time_of_day": "Après-midi",
    "crowd_density": "dense"
})

print(f"Niveau suspicion: {decision.suspicion_level}")
print(f"Recommandation: {decision.recommendation}")
```

### Gestion d'Erreurs

```python
try:
    vlm = create_vlm_model(ModelType.KIM)
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        print("⚠️ Mémoire GPU insuffisante, passage à SmolVLM")
        vlm = create_vlm_model(ModelType.SMOLVLM)
    else:
        raise e
```