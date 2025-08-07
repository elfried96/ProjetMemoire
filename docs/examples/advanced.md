# Exemples Avancés

## Surveillance Multi-Zones

### Analyse de Multiples Sections

```python
from src.orchestrator.controller import SurveillanceOrchestrator
from concurrent.futures import ThreadPoolExecutor
import asyncio

class MultiZoneSurveillance:
    def __init__(self):
        self.orchestrator = SurveillanceOrchestrator()
        self.zones = {
            "Entrée": {"priority": "high", "threshold": 0.7},
            "Caisses": {"priority": "critical", "threshold": 0.5},
            "Rayons": {"priority": "medium", "threshold": 0.8},
            "Sortie": {"priority": "high", "threshold": 0.6}
        }
    
    def analyze_all_zones(self, videos_by_zone):
        """Analyse parallèle de toutes les zones"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_zone = {
                executor.submit(
                    self.orchestrator.analyze,
                    videos_by_zone[zone],
                    zone,
                    "Après-midi",
                    "dense"
                ): zone for zone in self.zones
            }
            
            for future in future_to_zone:
                zone = future_to_zone[future]
                try:
                    result = future.result()
                    results[zone] = self._evaluate_zone_result(zone, result)
                except Exception as e:
                    print(f"❌ Erreur zone {zone}: {e}")
        
        return self._generate_global_report(results)
    
    def _evaluate_zone_result(self, zone, result):
        """Évaluation spécifique par zone"""
        config = self.zones[zone]
        
        # Ajustement du seuil par priorité de zone
        if result.confidence >= config["threshold"]:
            severity = self._calculate_severity(
                result.alert_level,
                config["priority"]
            )
            
            return {
                "result": result,
                "severity": severity,
                "requires_immediate_action": severity >= 8,
                "zone_specific_insights": self._get_zone_insights(zone)
            }
```

## Intelligence Contextuelle Avancée

### Apprentissage des Patterns Temporels

```python
from datetime import datetime, timedelta
import numpy as np

class TemporalPatternAnalyzer:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.temporal_patterns = {}
    
    def analyze_temporal_behavior(self, section, days_history=30):
        """Analyse les patterns temporels d'une section"""
        
        # Récupération historique
        historical_data = self.orchestrator.memory.get_historical_data(
            section, days=days_history
        )
        
        patterns = {
            "hourly_distribution": self._analyze_hourly_patterns(historical_data),
            "daily_trends": self._analyze_daily_trends(historical_data),
            "anomaly_hours": self._detect_anomaly_hours(historical_data),
            "peak_activity_prediction": self._predict_peak_hours(historical_data)
        }
        
        return patterns
    
    def _analyze_hourly_patterns(self, data):
        """Analyse des patterns horaires"""
        hourly_stats = {}
        
        for hour in range(24):
            hour_data = [d for d in data if d['timestamp'].hour == hour]
            
            if hour_data:
                hourly_stats[hour] = {
                    "avg_suspicion": np.mean([d['suspicion_score'] for d in hour_data]),
                    "alert_rate": len([d for d in hour_data if d['alert_level'] != 'low']) / len(hour_data),
                    "common_activities": self._extract_common_activities(hour_data),
                    "confidence": np.mean([d['confidence'] for d in hour_data])
                }
        
        return hourly_stats
    
    def predict_next_anomaly(self, section):
        """Prédiction de la prochaine anomalie probable"""
        patterns = self.analyze_temporal_behavior(section)
        current_hour = datetime.now().hour
        
        # Algorithme de prédiction basé sur l'historique
        risk_score = self._calculate_temporal_risk(patterns, current_hour)
        
        return {
            "predicted_time": self._estimate_next_high_risk_period(patterns),
            "risk_score": risk_score,
            "recommended_monitoring_level": self._get_monitoring_recommendation(risk_score)
        }
```

## Intégration Système de Sécurité

### Connecteur Universel

```python
import json
import requests
from abc import ABC, abstractmethod

class SecuritySystemConnector(ABC):
    @abstractmethod
    def send_alert(self, alert_data): pass
    
    @abstractmethod
    def get_system_status(self): pass

class GenericRESTConnector(SecuritySystemConnector):
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def send_alert(self, alert_data):
        """Envoi d'alerte vers système externe"""
        payload = {
            "timestamp": alert_data.timestamp.isoformat(),
            "severity": self._map_severity(alert_data.alert_level),
            "location": alert_data.section,
            "description": alert_data.recommendation,
            "confidence": alert_data.confidence,
            "ai_analysis": {
                "suspicion_level": alert_data.llm_decision.suspicion_level,
                "detected_objects": alert_data.vlm_analysis.detected_objects,
                "people_count": alert_data.vlm_analysis.people_count
            },
            "suggested_actions": self._generate_action_plan(alert_data)
        }
        
        response = requests.post(
            f"{self.base_url}/alerts",
            json=payload,
            headers=self.headers
        )
        
        return response.status_code == 200

class IntegratedSurveillanceSystem:
    def __init__(self, orchestrator, connector):
        self.orchestrator = orchestrator
        self.connector = connector
        self.active_alerts = {}
    
    def continuous_monitoring(self, video_streams):
        """Surveillance continue avec intégration système"""
        
        for stream_id, stream_config in video_streams.items():
            try:
                # Analyse en temps réel
                result = self.orchestrator.analyze(
                    stream_config["source"],
                    stream_config["section"],
                    self._get_current_time_period(),
                    stream_config["expected_density"]
                )
                
                # Évaluation du niveau de menace
                if self._requires_escalation(result):
                    self._escalate_alert(stream_id, result)
                
                # Mise à jour du statut de surveillance
                self._update_monitoring_status(stream_id, result)
                
            except Exception as e:
                self._handle_monitoring_error(stream_id, e)
    
    def _escalate_alert(self, stream_id, result):
        """Escalade d'alerte vers le système de sécurité"""
        
        # Prévention des alertes en double
        alert_key = f"{stream_id}_{result.timestamp.strftime('%Y%m%d_%H%M')}"
        
        if alert_key not in self.active_alerts:
            success = self.connector.send_alert(result)
            
            if success:
                self.active_alerts[alert_key] = {
                    "timestamp": result.timestamp,
                    "severity": result.alert_level,
                    "acknowledged": False
                }
                
                print(f"🚨 Alerte envoyée pour {stream_id}: {result.alert_level}")
            else:
                print(f"❌ Échec envoi alerte pour {stream_id}")
```

## Analyse Comportementale Avancée

### Détection d'Anomalies Comportementales

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np

class BehaviorAnalyzer:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.behavior_models = {}
        self.scaler = StandardScaler()
    
    def train_behavior_model(self, section, training_data):
        """Entraîne un modèle de comportement normal pour une section"""
        
        # Extraction des features comportementales
        features = self._extract_behavioral_features(training_data)
        
        # Normalisation
        features_scaled = self.scaler.fit_transform(features)
        
        # Clustering pour définir les comportements normaux
        clustering = DBSCAN(eps=0.5, min_samples=5)
        labels = clustering.fit_predict(features_scaled)
        
        self.behavior_models[section] = {
            "clusterer": clustering,
            "scaler": self.scaler,
            "normal_clusters": set(labels[labels >= 0]),  # Exclude noise (-1)
            "feature_names": self._get_feature_names()
        }
        
        print(f"✅ Modèle comportemental entraîné pour {section}")
        print(f"   Clusters normaux identifiés: {len(self.behavior_models[section]['normal_clusters'])}")
    
    def _extract_behavioral_features(self, data):
        """Extraction de features comportementales"""
        features = []
        
        for observation in data:
            feature_vector = [
                observation.vlm_analysis.people_count,
                len(observation.vlm_analysis.detected_objects),
                observation.vlm_analysis.confidence,
                observation.llm_decision.priority_score,
                self._calculate_movement_intensity(observation),
                self._calculate_interaction_score(observation),
                self._get_time_encoding(observation.timestamp),
                self._get_density_encoding(observation.crowd_density)
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def detect_behavioral_anomaly(self, current_analysis, section):
        """Détection d'anomalies comportementales en temps réel"""
        
        if section not in self.behavior_models:
            return {"anomaly_detected": False, "reason": "Modèle non entraîné"}
        
        model = self.behavior_models[section]
        
        # Extraction features de l'analyse courante
        current_features = self._extract_single_observation_features(current_analysis)
        current_features_scaled = model["scaler"].transform([current_features])
        
        # Prédiction du cluster
        predicted_cluster = model["clusterer"].fit_predict(current_features_scaled)[0]
        
        # Détection d'anomalie
        is_anomaly = predicted_cluster not in model["normal_clusters"]
        
        if is_anomaly:
            return {
                "anomaly_detected": True,
                "anomaly_score": self._calculate_anomaly_score(current_features_scaled, model),
                "behavioral_deviation": self._analyze_deviation(current_features, model),
                "recommended_action": self._get_anomaly_action(predicted_cluster)
            }
        
        return {"anomaly_detected": False}

    def generate_behavioral_report(self, section, period_days=7):
        """Génère un rapport comportemental détaillé"""
        
        recent_data = self.orchestrator.memory.get_historical_data(
            section, days=period_days
        )
        
        report = {
            "section": section,
            "analysis_period": f"{period_days} jours",
            "total_observations": len(recent_data),
            "behavioral_trends": self._analyze_trends(recent_data),
            "anomalies_detected": self._count_anomalies(recent_data, section),
            "recommendations": self._generate_behavioral_recommendations(recent_data, section)
        }
        
        return report
```

## Performance et Optimisation Avancées

### Auto-scaling selon la Charge

```python
import psutil
import threading
from queue import Queue

class AdaptiveProcessingManager:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.processing_queue = Queue()
        self.worker_threads = []
        self.active_workers = 2  # Nombre initial
        self.max_workers = 8
        self.performance_metrics = {}
        
    def auto_scale_processing(self):
        """Auto-scaling basé sur la charge système et les performances"""
        
        while True:
            # Monitoring des métriques système
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            queue_size = self.processing_queue.qsize()
            
            # Décision de scaling
            if queue_size > 10 and cpu_usage < 70 and self.active_workers < self.max_workers:
                self._scale_up()
            elif queue_size < 2 and self.active_workers > 1:
                self._scale_down()
            
            # Optimisation automatique des paramètres
            self._optimize_processing_parameters()
            
            time.sleep(30)  # Évaluation toutes les 30s
    
    def _optimize_processing_parameters(self):
        """Optimisation automatique des paramètres de traitement"""
        
        # Analyse des performances récentes
        if self.performance_metrics:
            avg_processing_time = np.mean([
                m['processing_time'] for m in self.performance_metrics[-50:]
            ])
            
            # Ajustement batch_size selon performance
            if avg_processing_time > 10:  # > 10 secondes
                if settings.config.batch_size > 1:
                    settings.config.batch_size -= 1
                    print(f"🔧 Réduction batch_size: {settings.config.batch_size}")
            
            elif avg_processing_time < 3:  # < 3 secondes
                if settings.config.batch_size < 6:
                    settings.config.batch_size += 1  
                    print(f"🔧 Augmentation batch_size: {settings.config.batch_size}")
    
    def intelligent_task_scheduling(self, tasks):
        """Planification intelligente des tâches selon priorité et ressources"""
        
        # Tri des tâches par priorité
        sorted_tasks = sorted(tasks, key=lambda t: (
            -self._get_priority_score(t),  # Priorité décroissante
            t.estimated_processing_time    # Temps croissant
        ))
        
        # Attribution optimale aux workers
        for task in sorted_tasks:
            optimal_worker = self._find_optimal_worker(task)
            optimal_worker.assign_task(task)
            
    def _get_priority_score(self, task):
        """Calcul du score de priorité d'une tâche"""
        base_score = 5
        
        # Facteurs d'augmentation de priorité
        if "caisse" in task.section.lower():
            base_score += 3
        if "sortie" in task.section.lower():
            base_score += 2
        if task.time_of_day in ["soir", "nuit"]:
            base_score += 1
        if task.crowd_density == "dense":
            base_score += 1
            
        return base_score
```