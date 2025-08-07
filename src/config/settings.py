"""
Configuration centralisée pour le projet Surveillance Orchestrator.
Permet de basculer facilement entre différents modèles et configurations.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ModelType(Enum):
    """Types de modèles VLM supportés."""
    SMOLVLM = "smolvlm"
    KIM = "kim"


@dataclass
class ModelConfig:
    """Configuration d'un modèle."""
    name: str
    model_id: str
    device: Optional[str] = None
    torch_dtype: str = "float16"
    trust_remote_code: bool = True
    enabled: bool = True


@dataclass
class ProcessingConfig:
    """Configuration du traitement vidéo."""
    seconds_per_frame: float = 2.0
    max_frames: int = 10
    target_size: tuple = (384, 384)
    contrast: float = 1.5
    brightness: float = 1.2
    denoise: bool = True


@dataclass
class SurveillanceConfig:
    """Configuration principale du système."""
    # Modèles
    primary_vlm: ModelType = ModelType.SMOLVLM
    fallback_vlm: ModelType = ModelType.KIM
    llm_model: str = "microsoft/phi-2"
    
    # Traitement
    processing: ProcessingConfig = None
    
    # Chemins
    outputs_dir: Path = None
    memory_dir: Path = None
    logs_dir: Path = None
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    
    # Performance
    batch_size: int = 4
    max_gpu_memory: float = 0.8  # Fraction de VRAM utilisable
    cleanup_after_analysis: bool = True
    
    def __post_init__(self):
        """Initialise les valeurs par défaut après création."""
        if self.processing is None:
            self.processing = ProcessingConfig()
        if self.outputs_dir is None:
            self.outputs_dir = Path("outputs")
        if self.memory_dir is None:
            self.memory_dir = Path("memory")
        if self.logs_dir is None:
            self.logs_dir = Path("outputs/logs")


# Configuration des modèles disponibles
MODEL_CONFIGS: Dict[ModelType, ModelConfig] = {
    ModelType.SMOLVLM: ModelConfig(
        name="SmolVLM",
        model_id="HuggingFaceTB/SmolVLM-Instruct",
        enabled=True
    ),
    ModelType.KIM: ModelConfig(
        name="KIM",
        model_id="microsoft/kosmos-2-patch14-224",  # Modèle existant similaire
        enabled=False  # Désactivé par défaut faute de ressources
    )
}


class Settings:
    """Gestionnaire de configuration singleton."""
    
    _instance = None
    _config: SurveillanceConfig = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = SurveillanceConfig()
        return cls._instance
    
    @property
    def config(self) -> SurveillanceConfig:
        return self._config
    
    def get_model_config(self, model_type: ModelType) -> ModelConfig:
        """Récupère la configuration d'un modèle."""
        return MODEL_CONFIGS[model_type]
    
    def get_primary_vlm_config(self) -> ModelConfig:
        """Récupère la configuration du VLM principal."""
        return self.get_model_config(self.config.primary_vlm)
    
    def set_primary_vlm(self, model_type: ModelType) -> None:
        """Change le VLM principal."""
        if not MODEL_CONFIGS[model_type].enabled:
            raise ValueError(f"Le modèle {model_type.value} n'est pas activé")
        self.config.primary_vlm = model_type
    
    def enable_model(self, model_type: ModelType, enabled: bool = True) -> None:
        """Active/désactive un modèle."""
        MODEL_CONFIGS[model_type].enabled = enabled
    
    def setup_directories(self) -> None:
        """Crée les répertoires nécessaires."""
        directories = [
            self.config.outputs_dir,
            self.config.memory_dir,
            self.config.logs_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def load_from_env(self) -> None:
        """Charge la configuration depuis les variables d'environnement."""
        # VLM principal
        primary_vlm = os.getenv("SURVEILLANCE_PRIMARY_VLM", "smolvlm").lower()
        if primary_vlm in [e.value for e in ModelType]:
            self.config.primary_vlm = ModelType(primary_vlm)
        
        # Niveau de log
        self.config.log_level = os.getenv("SURVEILLANCE_LOG_LEVEL", "INFO")
        
        # Taille de batch
        batch_size = os.getenv("SURVEILLANCE_BATCH_SIZE")
        if batch_size:
            self.config.batch_size = int(batch_size)
        
        # Mémoire GPU max
        max_gpu = os.getenv("SURVEILLANCE_MAX_GPU_MEMORY")
        if max_gpu:
            self.config.max_gpu_memory = float(max_gpu)


# Instance globale
settings = Settings()