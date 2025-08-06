"""
Modeles de surveillance intelligente.

Ce module contient les wrappers pour les differents modeles utilises:
- SmolVLM: Modele VLM principal (active par defaut)
- KIM: Modele VLM avance (desactive par defaut - necessite plus de ressources)
- Phi3: Modele LLM pour la prise de decision
"""

from typing import Union, Optional
import logging

from .base import BaseVLMModel, BaseLLMModel, ModelManager, AnalysisResult, SuspicionAnalysis
from .smolvlm_wrapper import SmolVLMWrapper
from .kim_wrapper import KIMWrapper
from .phi3_wrapper import Phi3Wrapper
from ..config import settings, ModelType

logger = logging.getLogger(__name__)


def create_vlm_model(
    model_type: Optional[ModelType] = None,
    device: Optional[str] = None
) -> BaseVLMModel:
    """
    Cree une instance du modele VLM configure.
    
    Args:
        model_type: Type de modele VLM a creer (utilise la config si None)
        device: Device a utiliser (auto-detection si None)
        
    Returns:
        Instance du modele VLM
        
    Raises:
        ValueError: Si le modele demande n'est pas disponible
    """
    if model_type is None:
        model_type = settings.config.primary_vlm
    
    model_config = settings.get_model_config(model_type)
    
    if not model_config.enabled:
        logger.warning(f"Modele {model_type.value} desactive, utilisation du fallback")
        # Utilise SmolVLM comme fallback
        if model_type != ModelType.SMOLVLM:
            return create_vlm_model(ModelType.SMOLVLM, device)
        else:
            raise ValueError("Aucun modele VLM disponible")
    
    if model_type == ModelType.SMOLVLM:
        return SmolVLMWrapper(model_config.model_id, device)
    elif model_type == ModelType.KIM:
        return KIMWrapper(model_config.model_id, device)
    else:
        raise ValueError(f"Type de modele VLM non supporte: {model_type}")


def create_llm_model(device: Optional[str] = None) -> BaseLLMModel:
    """
    Cree une instance du modele LLM pour la prise de decision.
    
    Args:
        device: Device a utiliser (auto-detection si None)
        
    Returns:
        Instance du modele LLM
    """
    return Phi3Wrapper(device=device)


def get_available_models() -> dict:
    """
    Retourne la liste des modeles disponibles.
    
    Returns:
        Dict avec les informations des modeles disponibles
    """
    models_info = {}
    
    for model_type in ModelType:
        config = settings.get_model_config(model_type)
        is_available = True
        
        if model_type == ModelType.KIM:
            is_available = KIMWrapper.is_available()
        
        models_info[model_type.value] = {
            "name": config.name,
            "model_id": config.model_id,
            "enabled": config.enabled,
            "available": is_available,
            "is_primary": model_type == settings.config.primary_vlm
        }
    
    return models_info


# Gestionnaire global de modeles
model_manager = ModelManager()

__all__ = [
    "BaseVLMModel",
    "BaseLLMModel", 
    "AnalysisResult",
    "SuspicionAnalysis",
    "SmolVLMWrapper",
    "KIMWrapper", 
    "Phi3Wrapper",
    "create_vlm_model",
    "create_llm_model",
    "get_available_models",
    "model_manager"
]