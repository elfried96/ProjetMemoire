"""
Orchestrateur principal de surveillance intelligente.
Coordonne l'analyse VLM, la prise de décision LLM et la gestion de la mémoire.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union

from ..models import create_vlm_model, create_llm_model, model_manager
from ..models.base import AnalysisResult, SuspicionAnalysis
from ..utils.preprocessing import video_processor
from ..utils.memory_optimizer import memory_monitor, memory_optimizer
from ..config import settings, ModelType
from ..utils.logging import get_surveillance_logger

logger = get_surveillance_logger()


class MemoryManager:
    """Gestionnaire de mémoire pour l'historique et les alertes."""
    
    def __init__(self, memory_path: Path):
        self.memory_path = memory_path
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.memory_path.exists():
            self._initialize_memory()
    
    def _initialize_memory(self) -> None:
        """Initialise le fichier de mémoire."""
        initial_data = {
            "events": [],
            "alerts": [],
            "statistics": {
                "total_analyses": 0,
                "total_alerts": 0,
                "last_maintenance": datetime.now().isoformat()
            }
        }
        self.memory_path.write_text(
            json.dumps(initial_data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        logger.info("Mémoire initialisée")
    
    def load_memory(self) -> dict:
        """Charge la mémoire depuis le fichier."""
        try:
            return json.loads(self.memory_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Erreur lecture mémoire: {e}")
            self._initialize_memory()
            return self.load_memory()
    
    def save_event(self, event: dict) -> None:
        """Sauvegarde un nouvel événement."""
        memory = self.load_memory()
        memory["events"].append(event)
        memory["statistics"]["total_analyses"] += 1
        
        # Nettoyage automatique (garde les 1000 derniers événements)
        if len(memory["events"]) > 1000:
            memory["events"] = memory["events"][-1000:]
            logger.debug("Nettoyage mémoire: événements anciens supprimés")
        
        self._save_memory(memory)
    
    def save_alert(self, alert: dict) -> None:
        """Sauvegarde une nouvelle alerte."""
        memory = self.load_memory()
        memory["alerts"].append(alert)
        memory["statistics"]["total_alerts"] += 1
        
        # Garde les alertes pendant 30 jours max
        cutoff_date = datetime.now() - timedelta(days=30)
        memory["alerts"] = [
            alert for alert in memory["alerts"]
            if datetime.fromisoformat(alert["timestamp"]) > cutoff_date
        ]
        
        self._save_memory(memory)
    
    def get_recent_alerts(self, section: str, limit: int = 5) -> List[str]:
        """Récupère les alertes récentes pour une section."""
        memory = self.load_memory()
        
        # Filtre par section et trie par date
        section_alerts = [
            alert for alert in memory["alerts"]
            if alert.get("section") == section
        ]
        section_alerts.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Formate les alertes récentes
        recent = section_alerts[:limit]
        return [
            f"{alert['decision']['alert_type']} ({alert['decision']['suspicion_level']}) - "
            f"{datetime.fromisoformat(alert['timestamp']).strftime('%H:%M')}"
            for alert in recent
        ]
    
    def get_statistics(self) -> dict:
        """Récupère les statistiques d'utilisation."""
        memory = self.load_memory()
        return memory.get("statistics", {})
    
    def _save_memory(self, memory: dict) -> None:
        """Sauvegarde la mémoire dans le fichier."""
        try:
            self.memory_path.write_text(
                json.dumps(memory, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except Exception as e:
            logger.error(f"Erreur sauvegarde mémoire: {e}")


class SurveillanceOrchestrator:
    """
    Orchestrateur principal de surveillance intelligente.
    
    Coordonne les analyses VLM, les décisions LLM et la gestion mémoire.
    Supporte le basculement automatique entre SmolVLM et KIM.
    """
    
    def __init__(self):
        """Initialise l'orchestrateur avec la configuration actuelle."""
        settings.setup_directories()
        
        # Auto-optimisation mémoire
        memory_optimizer.auto_configure_settings()
        
        # Gestionnaire de mémoire
        self.memory = MemoryManager(settings.config.memory_dir / "orchestration.json")
        
        # Logs de décisions
        self.decisions_log = settings.config.outputs_dir / "log_decisions.json"
        self._ensure_decisions_log()
        
        logger.info("🧠 Orchestrateur de surveillance initialisé")
        logger.info(f"VLM principal: {settings.config.primary_vlm.value}")
        memory_optimizer.log_memory_status("INIT")
        
        # Statistiques de session
        self.session_stats = {
            "analyses_count": 0,
            "alerts_count": 0,
            "start_time": datetime.now(),
            "model_switches": 0
        }
    
    def _ensure_decisions_log(self) -> None:
        """S'assure que le fichier de log des décisions existe."""
        if not self.decisions_log.exists():
            self.decisions_log.write_text("[]", encoding="utf-8")
    
    def analyze(
        self,
        video_path: Union[str, Path],
        section: str,
        time_of_day: str,
        crowd_density: str,
        use_keyframes: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Analyse complète d'une vidéo de surveillance.
        
        Args:
            video_path: Chemin vers la vidéo à analyser
            section: Section du magasin
            time_of_day: Moment de la journée
            crowd_density: Densité de la foule
            use_keyframes: Utiliser l'extraction de keyframes intelligente
            
        Returns:
            Résultat structuré de l'analyse ou None en cas d'erreur
        """
        try:
            with memory_monitor(f"Analyse {section}"):
                logger.info(f"🎯 Analyse de surveillance: {section} - {time_of_day}")
                analysis_start = datetime.now()
                
                # Vérification pression mémoire
                if memory_optimizer.check_memory_pressure():
                    memory_optimizer.aggressive_cleanup()
                
                # 1. Extraction des frames
                frames = self._extract_frames(video_path, use_keyframes)
                if not frames:
                    logger.error("Aucune frame extraite, abandon de l'analyse")
                    return None
                
                # 2. Analyse VLM
                vlm_result = self._analyze_with_vlm(frames, section, time_of_day, crowd_density)
                
                # 3. Prise de décision LLM
                decision = self._make_decision(vlm_result, section, time_of_day, crowd_density)
                
                # 4. Construction du résultat final
                result = self._build_final_result(
                    vlm_result, decision, section, analysis_start
                )
                
                # 5. Sauvegarde et logging
                self._save_analysis_result(result)
                
                # 6. Nettoyage mémoire si nécessaire
                if settings.config.cleanup_after_analysis:
                    self._cleanup_models()
                
                self.session_stats["analyses_count"] += 1
                if decision.suspicion_level in ["medium", "high"]:
                    self.session_stats["alerts_count"] += 1
                
                logger.info(f"✅ Analyse terminée en {(datetime.now() - analysis_start).total_seconds():.2f}s")
                return result
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'analyse: {e}")
            memory_optimizer.aggressive_cleanup()  # Nettoyage d'urgence
            return None
    
    def _extract_frames(self, video_path: Union[str, Path], use_keyframes: bool) -> List:
        """Extrait les frames de la vidéo."""
        try:
            if use_keyframes:
                logger.info("Extraction de keyframes intelligente")
                return video_processor.extract_keyframes(video_path)
            else:
                logger.info("Extraction de frames régulière")
                return video_processor.extract_frames(video_path)
        except Exception as e:
            logger.error(f"Erreur extraction frames: {e}")
            return []
    
    def _analyze_with_vlm(
        self, 
        frames: List, 
        section: str, 
        time_of_day: str, 
        crowd_density: str
    ) -> AnalysisResult:
        """Analyse les frames avec le modèle VLM."""
        try:
            # Création du modèle VLM
            vlm = create_vlm_model()
            
            # Analyse avec gestion d'erreur et fallback
            if len(frames) <= settings.config.batch_size:
                result = vlm.analyze_surveillance_scene(
                    frames, section, time_of_day, crowd_density
                )
            else:
                # Analyse par batch pour les grandes vidéos
                if hasattr(vlm, 'analyze_frames_batch'):
                    result = vlm.analyze_frames_batch(
                        frames, section, time_of_day, crowd_density
                    )
                else:
                    # Fallback: analyse du premier batch
                    batch = frames[:settings.config.batch_size]
                    result = vlm.analyze_surveillance_scene(
                        batch, section, time_of_day, crowd_density
                    )
                    logger.warning(f"VLM ne supporte pas l'analyse par batch, utilisé {len(batch)} frames")
            
            logger.info(f"Analyse VLM terminée (confiance: {result.confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Erreur analyse VLM: {e}")
            # Résultat d'erreur
            return AnalysisResult(
                summary=f"Erreur d'analyse VLM: {str(e)}",
                confidence=0.0,
                metadata={"error": True}
            )
    
    def _make_decision(
        self, 
        vlm_result: AnalysisResult, 
        section: str, 
        time_of_day: str, 
        crowd_density: str
    ) -> SuspicionAnalysis:
        """Prend une décision basée sur l'analyse VLM."""
        try:
            # Récupération des alertes récentes
            recent_alerts = self.memory.get_recent_alerts(section)
            
            # Construction du contexte pour le LLM
            context = {
                "section": section,
                "time": time_of_day,
                "density": crowd_density,
                "vlm_analysis": vlm_result.summary,
                "confidence": vlm_result.confidence,
                "last_alerts": recent_alerts
            }
            
            # Création et utilisation du modèle LLM
            llm = create_llm_model()
            decision = llm.analyze_context(context)
            
            logger.info(f"Décision LLM: {decision.alert_type} ({decision.suspicion_level})")
            return decision
            
        except Exception as e:
            logger.error(f"Erreur prise de décision: {e}")
            return SuspicionAnalysis(
                suspicion_level="unknown",
                alert_type="erreur_systeme",
                reasoning=f"Erreur technique: {str(e)}",
                action="verification_manuelle",
                confidence=0.0
            )
    
    def _build_final_result(
        self, 
        vlm_result: AnalysisResult, 
        decision: SuspicionAnalysis,
        section: str,
        analysis_start: datetime
    ) -> Dict[str, Any]:
        """Construit le résultat final de l'analyse."""
        return {
            "timestamp": analysis_start.isoformat(),
            "section": section,
            "thinking": vlm_result.thinking,
            "summary": vlm_result.summary,
            "decision": {
                "suspicion_level": decision.suspicion_level,
                "alert_type": decision.alert_type,
                "reasoning": decision.reasoning,
                "action": decision.action,
                "confidence": decision.confidence,
                "recommended_tools": decision.recommended_tools
            },
            "metadata": {
                "vlm_confidence": vlm_result.confidence,
                "llm_confidence": decision.confidence,
                "analysis_duration": (datetime.now() - analysis_start).total_seconds(),
                "model_used": vlm_result.metadata.get("model", "unknown") if vlm_result.metadata else "unknown"
            }
        }
    
    def _save_analysis_result(self, result: Dict[str, Any]) -> None:
        """Sauvegarde le résultat d'analyse."""
        # Sauvegarde en mémoire
        self.memory.save_event(result)
        
        # Sauvegarde des alertes importantes
        if result["decision"]["suspicion_level"] in ["medium", "high"]:
            self.memory.save_alert(result)
        
        # Log des décisions
        self._append_to_decisions_log(result)
    
    def _append_to_decisions_log(self, result: Dict[str, Any]) -> None:
        """Ajoute le résultat au log des décisions."""
        try:
            # Lecture du log existant
            if self.decisions_log.exists():
                existing_data = json.loads(self.decisions_log.read_text(encoding="utf-8"))
            else:
                existing_data = []
            
            # Ajout du nouveau résultat
            existing_data.append(result)
            
            # Limite à 500 dernières analyses
            if len(existing_data) > 500:
                existing_data = existing_data[-500:]
            
            # Sauvegarde
            self.decisions_log.write_text(
                json.dumps(existing_data, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde log décisions: {e}")
    
    def _cleanup_models(self) -> None:
        """Nettoie les modèles en mémoire."""
        try:
            model_manager.cleanup_inactive(keep_active=1)
            logger.debug("Nettoyage mémoire modèles effectué")
        except Exception as e:
            logger.warning(f"Erreur nettoyage modèles: {e}")
    
    def switch_to_kim(self) -> bool:
        """
        Bascule vers le modèle KIM si possible.
        
        Returns:
            True si le basculement a réussi
        """
        try:
            from ..models.kim_wrapper import KIMWrapper
            
            if KIMWrapper.is_available():
                settings.set_primary_vlm(ModelType.KIM)
                settings.enable_model(ModelType.KIM, True)
                self.session_stats["model_switches"] += 1
                logger.info("✅ Basculement vers KIM réussi")
                return True
            else:
                logger.warning("❌ KIM non disponible - ressources insuffisantes")
                return False
                
        except Exception as e:
            logger.error(f"Erreur basculement vers KIM: {e}")
            return False
    
    def switch_to_smolvlm(self) -> None:
        """Bascule vers SmolVLM (toujours disponible)."""
        settings.set_primary_vlm(ModelType.SMOLVLM)
        self.session_stats["model_switches"] += 1
        logger.info("✅ Basculement vers SmolVLM")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques de la session."""
        duration = datetime.now() - self.session_stats["start_time"]
        
        return {
            **self.session_stats,
            "session_duration": str(duration),
            "analyses_per_hour": round(
                self.session_stats["analyses_count"] / max(duration.total_seconds() / 3600, 0.1), 2
            ),
            "alert_rate": round(
                self.session_stats["alerts_count"] / max(self.session_stats["analyses_count"], 1) * 100, 2
            )
        }
    
    def cleanup(self) -> None:
        """Nettoyage complet de l'orchestrateur."""
        logger.info("🧹 Nettoyage de l'orchestrateur")
        model_manager.unload_all()
        
        # Affichage des stats finales
        stats = self.get_session_stats()
        logger.info(f"Session terminée: {stats['analyses_count']} analyses, {stats['alerts_count']} alertes")