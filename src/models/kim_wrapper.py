"""
Wrapper pour KIM - Modèle VLM avancé pour la surveillance (prêt mais désactivé).
À activer quand les ressources GPU seront disponibles.
"""

import torch
import gc
from typing import List, Optional, Dict, Any
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

from .base import BaseVLMModel, AnalysisResult
from ..config import settings
from ..utils.logging import get_surveillance_logger

logger = get_surveillance_logger()


class KIMWrapper(BaseVLMModel):
    """
    Wrapper pour le modèle KIM (désactivé par défaut).
    
    Note: Ce modèle nécessite plus de ressources GPU que SmolVLM.
    Activez-le uniquement si vous disposez d'une GPU avec au moins 8GB VRAM.
    """
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        if model_name is None:
            # Utilise un modèle Kosmos-2 comme placeholder pour KIM
            model_name = "microsoft/kosmos-2-patch14-224"
            
        super().__init__(model_name, device)
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Vérification des ressources avant chargement
        self._check_resources()
        
        if not settings.get_model_config(settings.ModelType.KIM).enabled:
            logger.info("KIM est désactivé dans la configuration")
            return
            
        # Chargement différé par défaut pour économiser les ressources
        if settings.config.cleanup_after_analysis:
            logger.info(f"KIM initialisé (chargement différé): {model_name}")
        else:
            logger.warning("KIM sera chargé immédiatement - vérifiez vos ressources GPU")
            self.load_model()
    
    def _check_resources(self) -> None:
        """Vérifie la disponibilité des ressources GPU."""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)
            
            if gpu_memory_gb < 6:
                logger.warning(
                    f"GPU avec {gpu_memory_gb:.1f}GB détectée. "
                    f"KIM recommande au moins 8GB VRAM."
                )
        else:
            logger.warning("Aucun GPU détecté. KIM sera très lent sur CPU.")
    
    def load_model(self) -> None:
        """Charge le modèle KIM en mémoire."""
        if self._is_loaded:
            return
            
        if not settings.get_model_config(settings.ModelType.KIM).enabled:
            raise RuntimeError(
                "KIM est désactivé. Activez-le avec settings.enable_model(ModelType.KIM)"
            )
            
        try:
            logger.info(f"Chargement de KIM: {self.model_name}")
            logger.warning("KIM nécessite beaucoup de VRAM - surveillez l'utilisation mémoire")
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                # Options d'optimisation mémoire
                load_in_8bit=True if self.device == "cuda" else False,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
            )
            
            if self.device != "auto" and not self.device == "cuda":
                self.model = self.model.to(self.device)
            
            self._is_loaded = True
            logger.info(f"✅ KIM chargé avec succès sur {self.device}")
            
            # Affiche l'utilisation mémoire
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / (1024**3)
                logger.info(f"Mémoire GPU utilisée: {memory_used:.2f}GB")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement de KIM: {e}")
            logger.info("💡 Conseil: Essayez SmolVLM si KIM échoue")
            raise
    
    def unload_model(self) -> None:
        """Décharge le modèle de la mémoire."""
        if not self._is_loaded:
            return
            
        try:
            logger.info("Déchargement de KIM...")
            del self.model
            del self.processor
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            
            self.model = None
            self.processor = None
            self._is_loaded = False
            logger.info("✅ KIM déchargé, mémoire libérée")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du déchargement: {e}")
    
    def analyze_images(
        self, 
        images: List[Image.Image], 
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> AnalysisResult:
        """
        Analyse une liste d'images avec KIM.
        
        Args:
            images: Liste d'images PIL à analyser
            prompt: Prompt textuel pour l'analyse
            max_new_tokens: Nombre maximum de tokens à générer
            temperature: Température pour la génération
            **kwargs: Arguments supplémentaires
            
        Returns:
            Résultat d'analyse structuré
        """
        self.ensure_loaded()
        
        if not images:
            logger.warning("Aucune image fournie pour l'analyse")
            return AnalysisResult()
        
        try:
            logger.info(f"Analyse de {len(images)} images avec KIM")
            
            inputs = self.processor(
                images=images,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    # Optimisations pour réduire l'usage mémoire
                    use_cache=False
                )

            # Extraire seulement les nouveaux tokens
            new_tokens = generated_ids[:, inputs.input_ids.shape[1]:]
            decoded = self.processor.batch_decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            thinking, summary = self._extract_thinking_and_summary(decoded)
            
            result = AnalysisResult(
                thinking=thinking,
                summary=summary,
                raw_output=decoded,
                confidence=self._estimate_confidence(decoded),
                metadata={
                    "model": "KIM",
                    "num_images": len(images),
                    "temperature": temperature,
                    "max_tokens": max_new_tokens
                }
            )
            
            logger.info("✅ Analyse KIM terminée")
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'analyse KIM: {e}")
            return AnalysisResult(
                summary=f"Erreur d'analyse KIM: {str(e)}",
                metadata={"error": True, "error_message": str(e)}
            )
    
    def _extract_thinking_and_summary(self, text: str, bot: str = "◁think▷", eot: str = "◁/think▷") -> tuple:
        """
        Extrait les sections de réflexion du texte généré par KIM.
        
        Args:
            text: Texte généré par le modèle
            bot: Balise de début de réflexion
            eot: Balise de fin de réflexion
            
        Returns:
            Tuple (thinking, summary)
        """
        if bot in text and eot in text:
            try:
                thinking_start = text.find(bot) + len(bot)
                thinking_end = text.find(eot)
                thinking = text[thinking_start:thinking_end].strip()
                summary = text[thinking_end + len(eot):].strip()
                return thinking, summary
            except Exception as e:
                logger.warning(f"Erreur extraction thinking/summary: {e}")
        
        # Fallback: tout le texte comme résumé
        return "", text.strip()
    
    def _estimate_confidence(self, text: str) -> float:
        """
        Estime un score de confiance basé sur le texte généré.
        KIM étant plus avancé, il peut fournir des indices de confiance explicites.
        
        Args:
            text: Texte généré
            
        Returns:
            Score de confiance entre 0 et 1
        """
        # Indicateurs de confiance spécifiques à KIM
        confidence_indicators = [
            ("très certain", 0.95), ("absolument", 0.95), ("définitivement", 0.9),
            ("clairement", 0.85), ("certainement", 0.8), ("probablement", 0.7),
            ("il semble", 0.6), ("possiblement", 0.5), ("peut-être", 0.4),
            ("incertain", 0.3), ("difficile à dire", 0.2)
        ]
        
        text_lower = text.lower()
        max_confidence = 0.6  # Confiance par défaut plus élevée pour KIM
        
        for indicator, conf in confidence_indicators:
            if indicator in text_lower:
                max_confidence = max(max_confidence, conf)
        
        return max_confidence
    
    def _build_surveillance_prompt(self, section: str, time_of_day: str, crowd_density: str) -> str:
        """Prompt spécialisé pour KIM avec plus de détails."""
        return f"""
Vous êtes un système de surveillance avancé utilisant l'intelligence artificielle KIM. Analysez ces images de magasin avec précision.

CONTEXTE DÉTAILLÉ:
- Section: {section}
- Heure: {time_of_day}
- Affluence: {crowd_density}

ANALYSE REQUISE:
1. Identification précise des personnes et objets
2. Analyse comportementale détaillée
3. Détection de patterns suspects
4. Évaluation des risques contextuels
5. Recommandations d'action

CRITÈRES DE SUSPICION AVANCÉS:
- Micro-expressions de stress ou nervosité
- Patterns de mouvement inhabituels
- Interactions sociales suspectes
- Manipulation d'objets non conforme
- Signaux visuels de dissimulation
- Comportements pré-criminels

INSTRUCTIONS SPÉCIALISÉES:
- Utilisez votre capacité d'analyse avancée
- Fournissez des détails précis et contextuels
- Indiquez votre niveau de certitude
- Suggérez des outils complémentaires si nécessaire

Format de réponse avec réflexion:
◁think▷
[Analyse détaillée étape par étape]
◁/think▷

[Résumé exécutif pour l'agent de sécurité]
        """.strip()
    
    def cleanup(self) -> None:
        """Alias pour unload_model pour compatibilité."""
        self.unload_model()
    
    @classmethod
    def is_available(cls) -> bool:
        """Vérifie si KIM peut être utilisé dans l'environnement actuel."""
        return (
            settings.get_model_config(settings.ModelType.KIM).enabled and
            torch.cuda.is_available() and
            torch.cuda.get_device_properties(0).total_memory > 6 * (1024**3)
        )