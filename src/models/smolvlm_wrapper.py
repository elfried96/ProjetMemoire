"""
Wrapper pour SmolVLM - Modèle VLM principal optimisé pour la surveillance.
Implémente l'interface BaseVLMModel avec gestion mémoire intelligente.
"""

import torch
import gc
from typing import List, Optional, Dict, Any
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

from .base import BaseVLMModel, AnalysisResult
from ..config import settings
from ..utils.logging import get_surveillance_logger

logger = get_surveillance_logger()


class SmolVLMWrapper(BaseVLMModel):
    """Wrapper pour le modèle SmolVLM optimisé pour la surveillance."""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        if model_name is None:
            model_config = settings.get_primary_vlm_config()
            model_name = model_config.model_id
            
        super().__init__(model_name, device)
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Chargement automatique si demandé
        if settings.config.cleanup_after_analysis:
            logger.info(f"SmolVLM initialisé (chargement différé): {model_name}")
        else:
            self.load_model()
    
    def load_model(self) -> None:
        """Charge le modèle SmolVLM en mémoire."""
        if self._is_loaded:
            return
            
        try:
            logger.info(f"Chargement de SmolVLM: {self.model_name}")
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device != "auto":
                self.model = self.model.to(self.device)
            
            self._is_loaded = True
            logger.info(f"✅ SmolVLM chargé avec succès sur {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement de SmolVLM: {e}")
            raise
    
    def unload_model(self) -> None:
        """Décharge le modèle de la mémoire."""
        if not self._is_loaded:
            return
            
        try:
            logger.info("Déchargement de SmolVLM...")
            del self.model
            del self.processor
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            self.model = None
            self.processor = None
            self._is_loaded = False
            logger.info("✅ SmolVLM déchargé, mémoire libérée")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du déchargement: {e}")
    
    def analyze_images(
        self, 
        images: List[Image.Image], 
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> AnalysisResult:
        """
        Analyse une liste d'images avec un prompt texte.
        
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
            logger.info(f"Analyse de {len(images)} images avec SmolVLM")
            
            # SmolVLM nécessite des tokens <image> dans le prompt
            image_tokens = "<image>" * len(images)
            formatted_prompt = f"{image_tokens}\n{prompt}"
            
            inputs = self.processor(
                images=images,
                text=formatted_prompt,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
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
                    "model": "SmolVLM",
                    "num_images": len(images),
                    "temperature": temperature,
                    "max_tokens": max_new_tokens
                }
            )
            
            logger.info("✅ Analyse SmolVLM terminée")
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'analyse SmolVLM: {e}")
            return AnalysisResult(
                summary=f"Erreur d'analyse: {str(e)}",
                metadata={"error": True, "error_message": str(e)}
            )
    
    def _extract_thinking_and_summary(self, text: str, bot: str = "◁think▷", eot: str = "◁/think▷") -> tuple:
        """
        Extrait les sections de réflexion du texte généré.
        
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
        
        Args:
            text: Texte généré
            
        Returns:
            Score de confiance entre 0 et 1
        """
        confidence_indicators = [
            ("certain", 0.9), ("sûr", 0.9), ("évident", 0.8),
            ("probable", 0.7), ("semble", 0.6), ("peut-être", 0.4),
            ("incertain", 0.3), ("difficile", 0.2)
        ]
        
        text_lower = text.lower()
        max_confidence = 0.5  # Confiance par défaut
        
        for indicator, conf in confidence_indicators:
            if indicator in text_lower:
                max_confidence = max(max_confidence, conf)
        
        return max_confidence
    
    def analyze_frames_batch(
        self, 
        frames: List[Image.Image], 
        section: str, 
        time_of_day: str, 
        crowd_density: str, 
        batch_size: Optional[int] = None
    ) -> AnalysisResult:
        """
        Analyse des frames par lot pour éviter l'overflow mémoire.

        Args:
            frames: Liste d'images PIL
            section: Zone du magasin
            time_of_day: Moment de la journée
            crowd_density: Affluence
            batch_size: Nombre d'images par lot (utilise config si None)
            
        Returns:
            Résultat d'analyse combiné
        """
        if not frames:
            logger.warning("Aucune frame fournie pour l'analyse")
            return AnalysisResult()
        
        if batch_size is None:
            batch_size = settings.config.batch_size
        
        logger.info(f"Analyse de {len(frames)} frames par lots de {batch_size}")
        
        all_results = []
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            logger.debug(f"Traitement du lot {i//batch_size + 1}/{(len(frames)-1)//batch_size + 1}")
            
            result = self.analyze_surveillance_scene(
                batch, section, time_of_day, crowd_density
            )
            all_results.append(result)
        
        # Combine les résultats
        combined_thinking = "\n\n--- BATCH ---\n\n".join(
            r.thinking for r in all_results if r.thinking
        )
        combined_summary = "\n".join(
            r.summary for r in all_results if r.summary
        )
        combined_raw = "\n--- BATCH ---\n".join(
            r.raw_output for r in all_results if r.raw_output
        )
        
        # Moyenne des scores de confiance
        confidences = [r.confidence for r in all_results if r.confidence > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return AnalysisResult(
            thinking=combined_thinking,
            summary=combined_summary,
            raw_output=combined_raw,
            confidence=avg_confidence,
            metadata={
                "model": "SmolVLM",
                "total_frames": len(frames),
                "batch_size": batch_size,
                "num_batches": len(all_results)
            }
        )
    
    def cleanup(self) -> None:
        """Alias pour unload_model pour compatibilité."""
        self.unload_model()