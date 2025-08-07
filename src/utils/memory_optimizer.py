"""
Optimiseur mémoire avancé pour Surveillance Orchestrator.
Outils pour surveiller et optimiser l'usage mémoire GPU/CPU.
"""

import gc
import psutil
import torch
from typing import Dict, Any, Optional
from contextlib import contextmanager
import logging

from ..config import settings
from .logging import get_surveillance_logger

logger = get_surveillance_logger()


class MemoryOptimizer:
    """Optimiseur mémoire avancé avec monitoring en temps réel."""
    
    def __init__(self):
        self.initial_memory = self._get_memory_info()
        self.peak_memory = self.initial_memory.copy()
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Récupère les informations mémoire actuelles."""
        info = {
            "cpu_percent": psutil.virtual_memory().percent,
            "cpu_available_gb": psutil.virtual_memory().available / (1024**3),
            "cpu_used_gb": psutil.virtual_memory().used / (1024**3)
        }
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_allocated = torch.cuda.memory_allocated()
            gpu_reserved = torch.cuda.memory_reserved()
            
            info.update({
                "gpu_total_gb": gpu_memory / (1024**3),
                "gpu_allocated_gb": gpu_allocated / (1024**3),
                "gpu_reserved_gb": gpu_reserved / (1024**3),
                "gpu_free_gb": (gpu_memory - gpu_reserved) / (1024**3),
                "gpu_utilization": (gpu_allocated / gpu_memory) * 100
            })
        
        return info
    
    def log_memory_status(self, stage: str = ""):
        """Log l'état actuel de la mémoire."""
        info = self._get_memory_info()
        
        logger.info(f"📊 Mémoire {stage}:")
        logger.info(f"  CPU: {info['cpu_percent']:.1f}% ({info['cpu_used_gb']:.1f}GB)")
        
        if "gpu_total_gb" in info:
            logger.info(f"  GPU: {info['gpu_utilization']:.1f}% ({info['gpu_allocated_gb']:.1f}GB / {info['gpu_total_gb']:.1f}GB)")
    
    def check_memory_pressure(self) -> bool:
        """Vérifie si le système est sous pression mémoire."""
        info = self._get_memory_info()
        
        cpu_pressure = info['cpu_percent'] > 80
        gpu_pressure = False
        
        if "gpu_utilization" in info:
            gpu_pressure = info['gpu_utilization'] > 85
        
        if cpu_pressure or gpu_pressure:
            logger.warning("⚠️ Pression mémoire détectée!")
            self.log_memory_status("PRESSURE")
            return True
        
        return False
    
    def aggressive_cleanup(self):
        """Nettoyage agressif de la mémoire."""
        logger.info("🧹 Nettoyage agressif de la mémoire...")
        
        # Nettoyage Python
        collected = gc.collect()
        logger.debug(f"Python GC: {collected} objets collectés")
        
        # Nettoyage GPU si disponible
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("Cache GPU vidé et synchronisé")
        
        # Force le nettoyage des imports inutiles
        self._cleanup_unused_modules()
        
        self.log_memory_status("APRÈS NETTOYAGE")
    
    def _cleanup_unused_modules(self):
        """Nettoie les modules Python inutiles."""
        import sys
        
        # Modules à conserver
        keep_modules = {
            'src', 'torch', 'transformers', 'PIL', 'cv2', 'numpy',
            'logging', 'json', 'pathlib', 'datetime', 'typing'
        }
        
        modules_to_remove = []
        for module_name in sys.modules.keys():
            if not any(keep in module_name for keep in keep_modules):
                if hasattr(sys.modules[module_name], '__file__'):
                    modules_to_remove.append(module_name)
        
        for module_name in modules_to_remove[:5]:  # Limite pour la sécurité
            try:
                del sys.modules[module_name]
            except:
                pass
    
    def optimize_for_gpu_size(self, gpu_memory_gb: float) -> Dict[str, Any]:
        """Optimise les paramètres selon la taille GPU."""
        if gpu_memory_gb < 4:
            return {
                "batch_size": 1,
                "use_8bit": True,
                "cleanup_after_analysis": True,
                "max_frames": 5,
                "gradient_checkpointing": True
            }
        elif gpu_memory_gb < 8:
            return {
                "batch_size": 2,
                "use_8bit": True,
                "cleanup_after_analysis": True,
                "max_frames": 8,
                "gradient_checkpointing": False
            }
        else:
            return {
                "batch_size": 4,
                "use_8bit": False,
                "cleanup_after_analysis": False,
                "max_frames": 15,
                "gradient_checkpointing": False
            }
    
    def auto_configure_settings(self):
        """Configure automatiquement les paramètres selon les ressources."""
        info = self._get_memory_info()
        
        if "gpu_total_gb" in info:
            gpu_gb = info["gpu_total_gb"]
            optimizations = self.optimize_for_gpu_size(gpu_gb)
            
            # Applique les optimisations
            settings.config.batch_size = optimizations["batch_size"]
            settings.config.cleanup_after_analysis = optimizations["cleanup_after_analysis"]
            settings.config.processing.max_frames = optimizations["max_frames"]
            
            logger.info(f"🎯 Auto-optimisation pour GPU {gpu_gb:.1f}GB:")
            logger.info(f"  Batch size: {optimizations['batch_size']}")
            logger.info(f"  Nettoyage auto: {optimizations['cleanup_after_analysis']}")
            logger.info(f"  Max frames: {optimizations['max_frames']}")
        
        else:
            # Mode CPU uniquement
            settings.config.batch_size = 1
            settings.config.cleanup_after_analysis = True
            settings.config.processing.max_frames = 5
            logger.info("🎯 Auto-optimisation pour CPU uniquement")


@contextmanager
def memory_monitor(stage: str = ""):
    """Context manager pour surveiller l'usage mémoire."""
    optimizer = MemoryOptimizer()
    
    logger.info(f"🚀 Début {stage}")
    optimizer.log_memory_status("AVANT")
    
    start_info = optimizer._get_memory_info()
    
    try:
        yield optimizer
    finally:
        end_info = optimizer._get_memory_info()
        optimizer.log_memory_status("APRÈS")
        
        # Calcul de la différence
        if "gpu_allocated_gb" in start_info and "gpu_allocated_gb" in end_info:
            gpu_diff = end_info["gpu_allocated_gb"] - start_info["gpu_allocated_gb"]
            logger.info(f"📈 Variation GPU: {gpu_diff:+.2f}GB")
        
        cpu_diff = end_info["cpu_used_gb"] - start_info["cpu_used_gb"]
        logger.info(f"📈 Variation CPU: {cpu_diff:+.2f}GB")


def optimize_model_loading(model_name: str, device: str = "auto") -> Dict[str, Any]:
    """Optimise les paramètres de chargement selon le modèle et les ressources."""
    optimizer = MemoryOptimizer()
    info = optimizer._get_memory_info()
    
    base_config = {
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "device_map": device,
        "trust_remote_code": True
    }
    
    if "gpu_total_gb" in info:
        gpu_gb = info["gpu_total_gb"]
        
        # Optimisations spécifiques aux modèles
        if "kim" in model_name.lower():
            if gpu_gb < 8:
                base_config.update({
                    "load_in_8bit": True,
                    "device_map": "auto",
                    "low_cpu_mem_usage": True
                })
            else:
                base_config.update({
                    "load_in_4bit": False,
                    "attn_implementation": "flash_attention_2"
                })
        
        elif "smol" in model_name.lower():
            # SmolVLM est déjà optimisé
            base_config.update({
                "low_cpu_mem_usage": True
            })
    
    return base_config


# Instance globale
memory_optimizer = MemoryOptimizer()