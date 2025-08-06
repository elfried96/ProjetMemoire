"""
Wrapper pour Phi-3 - Modèle LLM pour la prise de décision intelligente en surveillance.
Analyse les résultats VLM et prend des décisions contextuelles.
"""

import torch
import json
import gc
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseLLMModel, SuspicionAnalysis
from ..config import settings
from ..utils.logging import get_surveillance_logger

logger = get_surveillance_logger()


class Phi3Wrapper(BaseLLMModel):
    """Wrapper pour le modèle Phi-3 optimisé pour la prise de décision."""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        if model_name is None:
            model_name = settings.config.llm_model
            
        super().__init__(model_name, device)
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Chargement automatique si demandé
        if settings.config.cleanup_after_analysis:
            logger.info(f"Phi3 initialisé (chargement différé): {model_name}")
        else:
            self.load_model()
    
    def load_model(self) -> None:
        """Charge le modèle Phi-3 en mémoire."""
        if self._is_loaded:
            return
            
        try:
            logger.info(f"Chargement de Phi-3: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Ajout du pad_token si manquant
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device != "auto":
                self.model = self.model.to(self.device)
            
            self._is_loaded = True
            logger.info(f"✅ Phi-3 chargé avec succès sur {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement de Phi-3: {e}")
            raise
    
    def unload_model(self) -> None:
        """Décharge le modèle de la mémoire."""
        if not self._is_loaded:
            return
            
        try:
            logger.info("Déchargement de Phi-3...")
            del self.model
            del self.tokenizer
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            self.model = None
            self.tokenizer = None
            self._is_loaded = False
            logger.info("✅ Phi-3 déchargé, mémoire libérée")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du déchargement: {e}")
    
    def build_prompt(self, context: Dict[str, Any]) -> str:
        """
        Construit un prompt structuré pour l'analyse de surveillance.
        
        Args:
            context: Contexte d'analyse avec données VLM
            
        Returns:
            Prompt formaté pour Phi-3
        """
        vlm_analysis = context.get('vlm_analysis', 'Aucune analyse VLM disponible')
        section = context.get('section', 'Zone non spécifiée')
        time_of_day = context.get('time', 'Heure non spécifiée')
        crowd_density = context.get('density', 'Affluence inconnue')
        last_alerts = context.get('last_alerts', [])
        confidence = context.get('confidence', 0.5)
        
        recent_alerts = ', '.join(last_alerts) if last_alerts else 'Aucune alerte récente'
        
        return f"""
Vous êtes un système d'intelligence stratégique de surveillance. Analysez la situation et prenez une décision.

CONTEXTE OPÉRATIONNEL:
- Section surveillée: {section}
- Moment: {time_of_day}
- Affluence: {crowd_density}
- Confiance de l'analyse visuelle: {confidence:.2f}

ANALYSE VISUELLE (VLM):
{vlm_analysis}

HISTORIQUE RÉCENT:
{recent_alerts}

MISSION:
Évaluez la situation et fournissez une réponse JSON structurée UNIQUEMENT, sans autre texte.

CRITÈRES D'ÉVALUATION:
- Niveau de suspicion: low (aucun risque), medium (surveillance renforcée), high (intervention requise)
- Type d'alerte: rien, observation, dissimulation, repérage, tentative_vol, comportement_suspect
- Action: rien, surveiller_discretement, alerter_agent, intervenir, demander_renfort

FORMAT DE RÉPONSE (JSON uniquement):
{{
  "suspicion_level": "low|medium|high",
  "alert_type": "rien|observation|dissimulation|repérage|tentative_vol|comportement_suspect",
  "reasoning": "Explication concise de votre évaluation",
  "action": "rien|surveiller_discretement|alerter_agent|intervenir|demander_renfort",
  "confidence": 0.85,
  "priority": "low|medium|high|critical",
  "recommended_tools": ["detection_objets", "analyse_mouvement", "reconnaissance_faciale"],
  "estimated_risk": "Description du risque potentiel"
}}
        """.strip()
    
    def analyze_context(self, context: Dict[str, Any]) -> SuspicionAnalysis:
        """
        Analyse un contexte et prend une décision de surveillance.
        
        Args:
            context: Contexte d'analyse avec les données VLM
            
        Returns:
            Analyse de suspicion structurée
        """
        self.ensure_loaded()
        
        try:
            prompt = self.build_prompt(context)
            logger.info("Analyse contextuelle avec Phi-3")
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                max_length=2048,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.3,  # Plus déterministe pour les décisions
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )

            # Extraire seulement la réponse générée
            new_tokens = output_ids[:, inputs.input_ids.shape[1]:]
            decoded = self.tokenizer.decode(
                new_tokens[0], 
                skip_special_tokens=True
            ).strip()

            # Extraction et validation du JSON
            analysis = self._extract_and_validate_json(decoded)
            logger.info(f"✅ Décision Phi-3: {analysis.alert_type} ({analysis.suspicion_level})")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'analyse contextuelle: {e}")
            return SuspicionAnalysis(
                suspicion_level="unknown",
                alert_type="erreur_analyse",
                reasoning=f"Erreur technique: {str(e)}",
                action="verification_manuelle",
                confidence=0.0,
                recommended_tools=["verification_manuelle"]
            )
    
    def _extract_and_validate_json(self, text: str) -> SuspicionAnalysis:
        """
        Extrait et valide la réponse JSON de Phi-3.
        
        Args:
            text: Texte généré par le modèle
            
        Returns:
            Analyse de suspicion structurée
        """
        try:
            # Recherche du JSON dans le texte
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("Aucun JSON trouvé dans la réponse")
            
            json_str = text[json_start:json_end]
            data = json.loads(json_str)
            
            # Validation et normalisation des champs
            suspicion_level = self._validate_field(
                data.get("suspicion_level", "medium"),
                ["low", "medium", "high"],
                "medium"
            )
            
            alert_type = self._validate_field(
                data.get("alert_type", "observation"),
                ["rien", "observation", "dissimulation", "repérage", 
                 "tentative_vol", "comportement_suspect"],
                "observation"
            )
            
            action = self._validate_field(
                data.get("action", "surveiller_discretement"),
                ["rien", "surveiller_discretement", "alerter_agent", 
                 "intervenir", "demander_renfort"],
                "surveiller_discretement"
            )
            
            reasoning = str(data.get("reasoning", "Analyse automatique"))
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
            recommended_tools = data.get("recommended_tools", [])
            
            if not isinstance(recommended_tools, list):
                recommended_tools = []
            
            return SuspicionAnalysis(
                suspicion_level=suspicion_level,
                alert_type=alert_type,
                reasoning=reasoning,
                action=action,
                confidence=confidence,
                recommended_tools=recommended_tools
            )
            
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            logger.warning(f"Erreur extraction JSON: {e}")
            logger.debug(f"Texte problématique: {text[:200]}...")
            
            # Analyse de fallback basée sur des mots-clés
            return self._fallback_analysis(text)
    
    def _validate_field(self, value: str, valid_options: list, default: str) -> str:
        """Valide un champ contre une liste d'options valides."""
        return value.lower() if value.lower() in valid_options else default
    
    def _fallback_analysis(self, text: str) -> SuspicionAnalysis:
        """
        Analyse de fallback basée sur des mots-clés quand le JSON échoue.
        
        Args:
            text: Texte à analyser
            
        Returns:
            Analyse de suspicion basique
        """
        text_lower = text.lower()
        
        # Détection du niveau de suspicion par mots-clés
        if any(word in text_lower for word in ["urgent", "critique", "danger", "vol"]):
            suspicion_level = "high"
            action = "alerter_agent"
        elif any(word in text_lower for word in ["suspect", "bizarre", "inhabituel"]):
            suspicion_level = "medium"
            action = "surveiller_discretement"
        else:
            suspicion_level = "low"
            action = "rien"
        
        # Détection du type d'alerte
        if "dissimulation" in text_lower or "cacher" in text_lower:
            alert_type = "dissimulation"
        elif "repérage" in text_lower or "surveillance" in text_lower:
            alert_type = "repérage"
        elif "vol" in text_lower:
            alert_type = "tentative_vol"
        else:
            alert_type = "observation"
        
        return SuspicionAnalysis(
            suspicion_level=suspicion_level,
            alert_type=alert_type,
            reasoning=f"Analyse basique: {text[:100]}...",
            action=action,
            confidence=0.3,  # Confiance réduite pour le fallback
            recommended_tools=["verification_manuelle"]
        )
    
    def cleanup(self) -> None:
        """Alias pour unload_model pour compatibilité."""
        self.unload_model()