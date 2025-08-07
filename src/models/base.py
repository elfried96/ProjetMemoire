"""
Classes de base pour les modèles VLM et LLM.
Définit les interfaces communes pour faciliter l'interchangeabilité.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import torch
from PIL import Image


@dataclass
class AnalysisResult:
    """Résultat d'analyse d'un modèle."""
    thinking: str = ""
    summary: str = ""
    raw_output: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SuspicionAnalysis:
    """Résultat d'analyse de suspicion."""
    suspicion_level: str  # "low", "medium", "high"
    alert_type: str
    reasoning: str
    action: str
    confidence: float = 0.0
    recommended_tools: List[str] = None
    
    def __post_init__(self):
        if self.recommended_tools is None:
            self.recommended_tools = []


class BaseVLMModel(ABC):
    """Interface de base pour les modèles Vision-Language."""
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._is_loaded = False
    
    @abstractmethod
    def load_model(self) -> None:
        """Charge le modèle en mémoire."""
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Décharge le modèle de la mémoire."""
        pass
    
    @abstractmethod
    def analyze_images(
        self, 
        images: List[Image.Image], 
        prompt: str,
        **kwargs
    ) -> AnalysisResult:
        """
        Analyse une liste d'images avec un prompt.
        
        Args:
            images: Liste d'images PIL
            prompt: Prompt textuel pour l'analyse
            **kwargs: Arguments supplémentaires spécifiques au modèle
            
        Returns:
            Résultat d'analyse structuré
        """
        pass
    
    def analyze_surveillance_scene(
        self,
        images: List[Image.Image],
        section: str,
        time_of_day: str,
        crowd_density: str,
        **kwargs
    ) -> AnalysisResult:
        """
        Analyse spécialisée pour la surveillance.
        
        Args:
            images: Images à analyser
            section: Section du magasin
            time_of_day: Moment de la journée
            crowd_density: Densité de la foule
            
        Returns:
            Résultat d'analyse de surveillance
        """
        prompt = self._build_surveillance_prompt(section, time_of_day, crowd_density)
        return self.analyze_images(images, prompt, **kwargs)
    
    def _build_surveillance_prompt(self, section: str, time_of_day: str, crowd_density: str) -> str:
        """Construit le prompt de surveillance standard."""
        return f"""
Vous êtes un système de surveillance intelligente. Analysez ces images de magasin.

CONTEXTE:
- Section: {section}
- Heure: {time_of_day}
- Affluence: {crowd_density}

TÂCHES:
1. Identifiez toutes les personnes et leurs actions
2. Détectez tout comportement inhabituel ou suspect
3. Évaluez le niveau de risque

CRITÈRES SUSPECTS:
- Dissimulation d'objets
- Regards furtifs répétés
- Mouvements vers les sorties sans passer en caisse
- Manipulation d'étiquettes/emballages
- Comportement nerveux ou agité

INSTRUCTIONS:
- Soyez précis et objectif
- Indiquez votre niveau de confiance
- Suggérez des actions si nécessaire

◁think▷
""".strip()
    
    @property
    def is_loaded(self) -> bool:
        """Vérifie si le modèle est chargé."""
        return self._is_loaded
    
    def ensure_loaded(self) -> None:
        """S'assure que le modèle est chargé."""
        if not self._is_loaded:
            self.load_model()


class BaseLLMModel(ABC):
    """Interface de base pour les modèles de langage."""
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
    
    @abstractmethod
    def load_model(self) -> None:
        """Charge le modèle en mémoire."""
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Décharge le modèle de la mémoire."""
        pass
    
    @abstractmethod
    def analyze_context(self, context: Dict[str, Any]) -> SuspicionAnalysis:
        """
        Analyse un contexte et prend une décision.
        
        Args:
            context: Contexte d'analyse avec les données VLM
            
        Returns:
            Analyse de suspicion structurée
        """
        pass
    
    @property
    def is_loaded(self) -> bool:
        """Vérifie si le modèle est chargé."""
        return self._is_loaded
    
    def ensure_loaded(self) -> None:
        """S'assure que le modèle est chargé."""
        if not self._is_loaded:
            self.load_model()


class ModelManager:
    """Gestionnaire de modèles avec chargement/déchargement automatique."""
    
    def __init__(self):
        self._models: Dict[str, Union[BaseVLMModel, BaseLLMModel]] = {}
        self._active_models: List[str] = []
    
    def register_model(self, name: str, model: Union[BaseVLMModel, BaseLLMModel]) -> None:
        """Enregistre un modèle."""
        self._models[name] = model
    
    def get_model(self, name: str) -> Union[BaseVLMModel, BaseLLMModel]:
        """Récupère un modèle et s'assure qu'il est chargé."""
        if name not in self._models:
            raise ValueError(f"Modèle '{name}' non trouvé")
        
        model = self._models[name]
        model.ensure_loaded()
        
        if name not in self._active_models:
            self._active_models.append(name)
        
        return model
    
    def unload_model(self, name: str) -> None:
        """Décharge un modèle spécifique."""
        if name in self._models and self._models[name].is_loaded:
            self._models[name].unload_model()
            if name in self._active_models:
                self._active_models.remove(name)
    
    def unload_all(self) -> None:
        """Décharge tous les modèles actifs."""
        for name in self._active_models.copy():
            self.unload_model(name)
    
    def cleanup_inactive(self, keep_active: int = 2) -> None:
        """Décharge les modèles inactifs pour libérer la mémoire."""
        if len(self._active_models) > keep_active:
            models_to_unload = self._active_models[:-keep_active]
            for name in models_to_unload:
                self.unload_model(name)