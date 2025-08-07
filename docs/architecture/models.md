# Architecture des Modèles

## Vue d'ensemble

Le système utilise deux types de modèles AI qui collaborent :

- **VLM (Vision-Language Models)** : Analyse des images/vidéos
- **LLM (Language Models)** : Prise de décision contextuelle

## Modèles VLM

### SmolVLM (Principal)
- **Usage** : Modèle par défaut, optimisé ressources limitées
- **VRAM** : ~2-4GB
- **Performance** : Rapide, efficace
- **Cas d'usage** : Surveillance générale, CPU/GPU limité

```python
from src.models import create_vlm_model
vlm = create_vlm_model()  # SmolVLM par défaut
```

### KIM (Avancé)  
- **Usage** : Modèle avancé pour analyses complexes
- **VRAM** : ~6-8GB
- **Performance** : Plus lent mais plus précis
- **Cas d'usage** : Analyses critiques, GPU puissant

```python
from src.models.kim_wrapper import KIMWrapper
if KIMWrapper.is_available():
    vlm = create_vlm_model(ModelType.KIM)
```

## Modèle LLM

### Phi-3 (Décision)
- **Usage** : Prise de décision contextuelle
- **VRAM** : ~1-2GB  
- **Rôle** : Analyse les résultats VLM et décide des actions

## Interface Commune

Tous les modèles implémentent les interfaces de base :

```python
class BaseVLMModel(ABC):
    def analyze_images(self, images, prompt) -> AnalysisResult
    def load_model(self) -> None
    def unload_model(self) -> None

class BaseLLMModel(ABC):
    def analyze_context(self, context) -> SuspicionAnalysis
```

## Basculement Automatique

Le système peut basculer automatiquement entre modèles selon :

- Ressources GPU disponibles
- Complexité de la tâche  
- Performance historique
- Configuration utilisateur