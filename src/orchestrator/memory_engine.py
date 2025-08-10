"""
Moteur de mémoire contextuelle avancé pour l'orchestrateur de surveillance.
Gère l'apprentissage, la reconnaissance de patterns et l'optimisation des décisions.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict, Counter
import statistics

from ..utils.logging import get_surveillance_logger

logger = get_surveillance_logger()


class PatternDetector:
    """Détecteur de patterns comportementaux."""
    
    def __init__(self):
        self.suspicious_patterns = []
        self.normal_patterns = []
        self.learning_threshold = 5  # Minimum d'occurrences pour apprendre
    
    def add_observation(self, context: Dict[str, Any], decision: Dict[str, Any]):
        """Ajoute une observation pour l'apprentissage de patterns."""
        pattern = {
            "section": context.get("section"),
            "time": context.get("time"),
            "density": context.get("density"),
            "suspicion_level": decision.get("suspicion_level"),
            "alert_type": decision.get("alert_type"),
            "confidence": decision.get("confidence", 0.0)
        }
        
        if decision.get("suspicion_level") in ["medium", "high"]:
            self.suspicious_patterns.append(pattern)
        else:
            self.normal_patterns.append(pattern)
    
    def get_similar_patterns(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Trouve des patterns similaires au contexte donné."""
        similar = []
        
        for pattern in self.suspicious_patterns:
            similarity_score = self._calculate_similarity(context, pattern)
            if similarity_score > 0.7:  # 70% de similarité minimum
                pattern["similarity"] = similarity_score
                similar.append(pattern)
        
        return sorted(similar, key=lambda x: x["similarity"], reverse=True)
    
    def _calculate_similarity(self, context: Dict[str, Any], pattern: Dict[str, Any]) -> float:
        """Calcule la similarité entre un contexte et un pattern."""
        matches = 0
        total = 0
        
        comparable_fields = ["section", "time", "density"]
        
        for field in comparable_fields:
            if field in context and field in pattern:
                total += 1
                if context[field] == pattern[field]:
                    matches += 1
        
        return matches / total if total > 0 else 0.0


class ContextualLearning:
    """Système d'apprentissage contextuel."""
    
    def __init__(self):
        self.section_statistics = defaultdict(lambda: {
            "total_analyses": 0,
            "alerts": Counter(),
            "confidence_scores": [],
            "time_patterns": Counter(),
            "density_patterns": Counter()
        })
        
        self.model_performance = defaultdict(lambda: {
            "accuracy_estimates": [],
            "confidence_trends": [],
            "processing_times": []
        })
    
    def update_section_stats(self, section: str, decision: Dict[str, Any], metadata: Dict[str, Any]):
        """Met à jour les statistiques d'une section."""
        stats = self.section_statistics[section]
        
        stats["total_analyses"] += 1
        stats["alerts"][decision.get("alert_type", "unknown")] += 1
        stats["confidence_scores"].append(decision.get("confidence", 0.0))
        
        if "analysis_duration" in metadata:
            # Pas de processing_times ici, c'est pour les modèles
            pass
    
    def update_model_performance(self, model_name: str, confidence: float, processing_time: float):
        """Met à jour les performances d'un modèle."""
        perf = self.model_performance[model_name]
        
        perf["confidence_trends"].append(confidence)
        perf["processing_times"].append(processing_time)
        
        # Garde seulement les 100 dernières mesures
        if len(perf["confidence_trends"]) > 100:
            perf["confidence_trends"] = perf["confidence_trends"][-100:]
        if len(perf["processing_times"]) > 100:
            perf["processing_times"] = perf["processing_times"][-100:]
    
    def get_section_insights(self, section: str) -> Dict[str, Any]:
        """Récupère des insights sur une section."""
        stats = self.section_statistics[section]
        
        if stats["total_analyses"] == 0:
            return {"status": "no_data"}
        
        # Convertir alerts en Counter si c'est un dict (après chargement JSON)
        alerts = stats["alerts"]
        if isinstance(alerts, dict) and not isinstance(alerts, Counter):
            alerts = Counter(alerts)
            stats["alerts"] = alerts
        
        insights = {
            "total_analyses": stats["total_analyses"],
            "most_common_alert": alerts.most_common(1)[0] if alerts else ("none", 0),
            "average_confidence": statistics.mean(stats["confidence_scores"]) if stats["confidence_scores"] else 0.0,
            "alert_rate": sum(alerts.values()) / stats["total_analyses"] * 100 if stats["total_analyses"] > 0 else 0
        }
        
        return insights
    
    def recommend_model_for_context(self, context: Dict[str, Any]) -> Optional[str]:
        """Recommande le meilleur modèle pour un contexte donné."""
        # Logique simple basée sur les performances passées
        best_model = None
        best_score = 0.0
        
        for model_name, perf in self.model_performance.items():
            if not perf["confidence_trends"]:
                continue
                
            # Score basé sur confiance moyenne et vitesse
            avg_confidence = statistics.mean(perf["confidence_trends"])
            avg_speed = statistics.mean(perf["processing_times"]) if perf["processing_times"] else 10.0
            
            # Score composite (privilégie la confiance, pénalise la lenteur)
            score = avg_confidence * 0.8 - (avg_speed / 10.0) * 0.2
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model


class IntelligentMemoryManager:
    """Gestionnaire de mémoire contextuelle intelligent."""
    
    def __init__(self, memory_path: Path):
        self.memory_path = memory_path
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Composants intelligents
        self.pattern_detector = PatternDetector()
        self.contextual_learning = ContextualLearning()
        
        # Cache en mémoire pour performance
        self._memory_cache = None
        self._last_cache_update = None
        self.cache_duration = timedelta(minutes=5)  # Cache valide 5 minutes
        
        if not self.memory_path.exists():
            self._initialize_memory()
        else:
            self._load_intelligent_data()
    
    def _initialize_memory(self) -> None:
        """Initialise la mémoire intelligente."""
        initial_data = {
            "events": [],
            "alerts": [],
            "patterns": {
                "suspicious": [],
                "normal": []
            },
            "learning": {
                "section_stats": {},
                "model_performance": {}
            },
            "statistics": {
                "total_analyses": 0,
                "total_alerts": 0,
                "learning_iterations": 0,
                "last_maintenance": datetime.now().isoformat(),
                "last_pattern_update": datetime.now().isoformat()
            }
        }
        
        self._save_memory(initial_data)
        logger.info("Mémoire intelligente initialisée")
    
    def _load_intelligent_data(self):
        """Charge les données intelligentes depuis le fichier."""
        try:
            memory = self.load_memory()
            
            # Charge les patterns
            if "patterns" in memory:
                self.pattern_detector.suspicious_patterns = memory["patterns"].get("suspicious", [])
                self.pattern_detector.normal_patterns = memory["patterns"].get("normal", [])
            
            # Charge l'apprentissage
            if "learning" in memory:
                learning_data = memory["learning"]
                
                if "section_stats" in learning_data:
                    for section, stats in learning_data["section_stats"].items():
                        self.contextual_learning.section_statistics[section] = stats
                
                if "model_performance" in learning_data:
                    for model, perf in learning_data["model_performance"].items():
                        self.contextual_learning.model_performance[model] = perf
            
            logger.info("Données intelligentes chargées")
            
        except Exception as e:
            logger.error(f"Erreur chargement données intelligentes: {e}")
    
    def load_memory(self) -> dict:
        """Charge la mémoire avec cache."""
        now = datetime.now()
        
        if (self._memory_cache is None or 
            self._last_cache_update is None or 
            now - self._last_cache_update > self.cache_duration):
            
            try:
                self._memory_cache = json.loads(self.memory_path.read_text(encoding="utf-8"))
                self._last_cache_update = now
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"Erreur lecture mémoire: {e}")
                self._initialize_memory()
                return self.load_memory()
        
        return self._memory_cache
    
    def save_intelligent_event(self, event: Dict[str, Any], context: Dict[str, Any]):
        """Sauvegarde un événement avec apprentissage intelligent."""
        memory = self.load_memory()
        
        # Sauvegarde l'événement standard
        memory["events"].append(event)
        memory["statistics"]["total_analyses"] += 1
        
        # Apprentissage des patterns
        decision = event.get("decision", {})
        self.pattern_detector.add_observation(context, decision)
        
        # Apprentissage contextuel
        section = event.get("section", "unknown")
        metadata = event.get("metadata", {})
        self.contextual_learning.update_section_stats(section, decision, metadata)
        
        # Apprentissage des performances modèles
        model_used = metadata.get("model_used", "unknown")
        if model_used != "unknown":
            confidence = decision.get("confidence", 0.0)
            processing_time = metadata.get("analysis_duration", 0.0)
            self.contextual_learning.update_model_performance(model_used, confidence, processing_time)
        
        # Sauvegarde des données intelligentes
        memory["patterns"] = {
            "suspicious": self.pattern_detector.suspicious_patterns[-50:],  # Garde les 50 derniers
            "normal": self.pattern_detector.normal_patterns[-50:]
        }
        
        memory["learning"] = {
            "section_stats": dict(self.contextual_learning.section_statistics),
            "model_performance": dict(self.contextual_learning.model_performance)
        }
        
        memory["statistics"]["learning_iterations"] += 1
        memory["statistics"]["last_pattern_update"] = datetime.now().isoformat()
        
        # Nettoyage automatique (garde les 1000 derniers événements)
        if len(memory["events"]) > 1000:
            memory["events"] = memory["events"][-1000:]
            logger.debug("Nettoyage mémoire: anciens événements supprimés")
        
        self._save_memory(memory)
        self._memory_cache = memory  # Met à jour le cache
        
        logger.debug(f"Événement intelligent sauvegardé pour {section}")
    
    def get_contextual_alerts(self, section: str, context: Dict[str, Any], limit: int = 5) -> List[str]:
        """Récupère des alertes contextuelles intelligentes."""
        # Alertes récentes classiques
        recent_alerts = self.get_recent_alerts(section, limit)
        
        # Patterns similaires détectés
        similar_patterns = self.pattern_detector.get_similar_patterns(context)
        
        # Insights de la section
        insights = self.contextual_learning.get_section_insights(section)
        
        # Combine tout en alertes contextuelles
        contextual_alerts = recent_alerts.copy()
        
        if similar_patterns:
            pattern_alert = f"Pattern suspect similaire détecté (confiance: {similar_patterns[0]['confidence']:.2f})"
            contextual_alerts.append(pattern_alert)
        
        if insights.get("alert_rate", 0) > 50:
            contextual_alerts.append(f"Zone à haut risque ({insights['alert_rate']:.1f}% d'alertes)")
        
        return contextual_alerts[:limit]
    
    def get_recent_alerts(self, section: str, limit: int = 3) -> List[str]:
        """Version basique des alertes récentes."""
        memory = self.load_memory()
        
        # Filtre par section et trie par date
        section_alerts = [
            alert for alert in memory.get("alerts", [])
            if alert.get("section") == section
        ]
        section_alerts.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Formate les alertes récentes
        recent = section_alerts[:limit]
        return [
            f"{alert['decision']['alert_type']} ({alert['decision']['suspicion_level']}) - "
            f"{datetime.fromisoformat(alert['timestamp']).strftime('%H:%M')}"
            for alert in recent
            if 'decision' in alert and 'timestamp' in alert
        ]
    
    def get_model_recommendation(self, context: Dict[str, Any]) -> Optional[str]:
        """Recommande le meilleur modèle pour le contexte."""
        return self.contextual_learning.recommend_model_for_context(context)
    
    def get_intelligence_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques d'intelligence."""
        memory = self.load_memory()
        
        return {
            "total_patterns": len(self.pattern_detector.suspicious_patterns) + len(self.pattern_detector.normal_patterns),
            "suspicious_patterns": len(self.pattern_detector.suspicious_patterns),
            "sections_learned": len(self.contextual_learning.section_statistics),
            "models_tracked": len(self.contextual_learning.model_performance),
            "learning_iterations": memory.get("statistics", {}).get("learning_iterations", 0),
            "last_pattern_update": memory.get("statistics", {}).get("last_pattern_update", "N/A")
        }
    
    def _save_memory(self, memory: dict) -> None:
        """Sauvegarde la mémoire."""
        try:
            self.memory_path.write_text(
                json.dumps(memory, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except Exception as e:
            logger.error(f"Erreur sauvegarde mémoire: {e}")
    
    def optimize_memory_usage(self):
        """Optimise l'usage de la mémoire contextuelle."""
        memory = self.load_memory()
        
        # Nettoie les anciens événements (garde 500 derniers)
        if len(memory.get("events", [])) > 500:
            memory["events"] = memory["events"][-500:]
        
        # Nettoie les anciennes alertes (30 derniers jours)
        cutoff_date = datetime.now() - timedelta(days=30)
        memory["alerts"] = [
            alert for alert in memory.get("alerts", [])
            if datetime.fromisoformat(alert["timestamp"]) > cutoff_date
        ]
        
        # Limite les patterns (50 de chaque type)
        if len(self.pattern_detector.suspicious_patterns) > 50:
            self.pattern_detector.suspicious_patterns = self.pattern_detector.suspicious_patterns[-50:]
        
        if len(self.pattern_detector.normal_patterns) > 50:
            self.pattern_detector.normal_patterns = self.pattern_detector.normal_patterns[-50:]
        
        # Met à jour la mémoire
        memory["patterns"] = {
            "suspicious": self.pattern_detector.suspicious_patterns,
            "normal": self.pattern_detector.normal_patterns
        }
        
        self._save_memory(memory)
        self._memory_cache = None  # Invalide le cache
        
        logger.info("Optimisation mémoire contextuelle terminée")