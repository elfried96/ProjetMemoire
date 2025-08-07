# API Orchestrateur

## Module `src.orchestrator.controller`

L'orchestrateur principal coordonne tous les composants du système pour fournir une analyse intelligente de surveillance.

## Classe Principale

### `SurveillanceOrchestrator`

Orchestrateur central du système de surveillance intelligente.

```python
from src.orchestrator.controller import SurveillanceOrchestrator

orchestrator = SurveillanceOrchestrator(
    memory_path="data/memory.json",
    auto_optimize=True
)
```

**Paramètres d'initialisation** :
- `memory_path` : Chemin vers le fichier de mémoire contextuelle
- `auto_optimize` : Activation des optimisations automatiques

## Méthodes Principales

### `analyze()`

Effectue une analyse complète de surveillance vidéo.

```python
def analyze(
    self,
    video_path: str,
    section: str,
    time_of_day: str,
    crowd_density: str,
    use_keyframes: bool = False
) -> SurveillanceResult:
```

**Paramètres** :
- `video_path` : Chemin vers le fichier vidéo
- `section` : Zone de surveillance (ex: "Caisse principale")
- `time_of_day` : Moment ("Matin", "Après-midi", "Soirée", "Nuit")
- `crowd_density` : Densité ("faible", "moyenne", "dense")
- `use_keyframes` : Utiliser la sélection intelligente de frames

**Retourne** : `SurveillanceResult` avec analyse complète

**Exemple** :
```python
result = orchestrator.analyze(
    "videos/surveillance_caisse.mp4",
    "Caisse principale",
    "Après-midi", 
    "dense",
    use_keyframes=True
)

print(f"Niveau d'alerte: {result.alert_level}")
print(f"Recommandation: {result.recommendation}")
```

### `get_section_insights()`

Récupère les statistiques intelligentes pour une section.

```python
def get_section_insights(self, section: str) -> Dict[str, Any]:
```

**Retourne** un dictionnaire avec :
- `total_analyses` : Nombre total d'analyses
- `alert_rate` : Taux d'alerte (%)
- `avg_confidence` : Confiance moyenne
- `most_common_alerts` : Types d'alertes fréquentes
- `patterns_detected` : Patterns détectés
- `recommendations` : Recommandations personnalisées

**Exemple** :
```python
insights = orchestrator.get_section_insights("Entrée principale")
print(f"Taux d'alerte: {insights['alert_rate']:.1f}%")
print(f"Patterns détectés: {len(insights['patterns_detected'])}")
```

### `get_session_stats()`

Statistiques de la session actuelle.

```python
def get_session_stats(self) -> Dict[str, Any]:
```

**Retourne** :
- `total_analyses` : Analyses effectuées
- `intelligence.total_patterns` : Patterns appris
- `intelligence.suspicious_patterns` : Patterns suspects
- `intelligence.sections_learned` : Sections analysées
- `performance.avg_processing_time` : Temps moyen de traitement

### `optimize_contextual_memory()`

Lance une optimisation de la mémoire contextuelle.

```python
def optimize_contextual_memory(self) -> Dict[str, int]:
```

**Actions d'optimisation** :
- Nettoyage des données obsolètes
- Compression des patterns similaires
- Réindexation des statistiques
- Optimisation du cache

**Exemple** :
```python
stats = orchestrator.optimize_contextual_memory()
print(f"Patterns supprimés: {stats['patterns_cleaned']}")
print(f"Événements archivés: {stats['events_archived']}")
```

### `switch_to_kim()`

Bascule vers le modèle KIM si les ressources le permettent.

```python
def switch_to_kim(self) -> bool:
```

**Retourne** : `True` si le basculement a réussi

**Exemple** :
```python
if orchestrator.switch_to_kim():
    print("✅ Basculement vers KIM réussi")
else:
    print("❌ Ressources insuffisantes pour KIM")
```

## Structures de Données

### `SurveillanceResult`

Résultat complet d'une analyse de surveillance.

```python
@dataclass
class SurveillanceResult:
    # Résultats principaux
    alert_level: str                    # "low", "medium", "high"
    confidence: float                   # Confiance globale (0.0-1.0)
    recommendation: str                 # Recommandation d'action
    
    # Détails de l'analyse
    vlm_analysis: AnalysisResult       # Résultat VLM détaillé
    llm_decision: SuspicionAnalysis    # Décision LLM
    
    # Contexte enrichi
    contextual_alerts: List[str]       # Alertes contextuelles
    learned_insights: List[str]        # Insights appris
    
    # Métadonnées
    processing_time: float             # Temps total de traitement
    frames_analyzed: int               # Nombre de frames analysées
    model_used: str                    # Modèle VLM utilisé
    timestamp: datetime               # Horodatage de l'analyse
    
    # Intelligence contextuelle
    pattern_matches: List[str]         # Patterns reconnus
    anomaly_score: float              # Score d'anomalie (0.0-1.0)
    historical_comparison: Dict       # Comparaison historique
```

## Gestion de la Mémoire Contextuelle

### `IntelligentMemoryManager`

Le gestionnaire de mémoire contextuelle permet à l'orchestrateur d'apprendre et de s'adapter.

```python
# Accès au gestionnaire de mémoire
memory = orchestrator.memory

# Ajout d'observation manuelle
memory.add_observation(
    context={"section": "Caisse 1", "time": "Après-midi"},
    decision={"suspicion_level": "medium", "confidence": 0.8}
)

# Récupération d'alertes contextuelles
alerts = memory.get_contextual_alerts("Caisse 1", {
    "time": "Après-midi",
    "density": "dense"
})
```

### Apprentissage Automatique

L'orchestrateur apprend automatiquement des patterns :

```python
# Pattern automatiquement détecté après plusieurs analyses
pattern_example = {
    "section": "Sortie",
    "time": "Soirée", 
    "density": "faible",
    "typical_suspicion": "low",
    "confidence": 0.85,
    "frequency": 15  # Observé 15 fois
}
```

## Optimisations et Performance

### Monitoring Automatique

```python
# L'orchestrateur monitore automatiquement :
with memory_monitor("Analyse complète"):
    result = orchestrator.analyze(video_path, section, time, density)
    # Logs automatiques de l'utilisation mémoire
```

### Configuration Adaptive

```python
# Configuration automatique selon les ressources
orchestrator._configure_for_resources()

# Configuration manuelle
orchestrator._vlm = create_vlm_model(ModelType.KIM)
orchestrator._llm = create_llm_model("microsoft/phi-3")
```

## Exemples d'Usage Avancés

### Analyse en Lot

```python
video_files = ["video1.mp4", "video2.mp4", "video3.mp4"]
results = []

for video in video_files:
    result = orchestrator.analyze(
        video, "Entrée", "Matin", "moyenne"
    )
    results.append(result)
    
    # Apprentissage continu
    if result.alert_level == "high":
        print(f"⚠️ Alerte détectée dans {video}")
```

### Surveillance Continue

```python
import time

def continuous_surveillance(video_stream):
    while True:
        frame_batch = video_stream.get_next_batch()
        
        result = orchestrator.analyze_frames(
            frame_batch, "Zone A", "temps_reel", "variable"
        )
        
        if result.alert_level in ["medium", "high"]:
            send_alert(result)
        
        time.sleep(30)  # Analyse toutes les 30 secondes
```

### Rapport Intelligent

```python
def generate_intelligence_report(orchestrator):
    """Génère un rapport d'intelligence du système"""
    
    stats = orchestrator.get_session_stats()
    
    report = {
        "analyses_effectuees": stats["total_analyses"],
        "patterns_appris": stats["intelligence"]["total_patterns"],
        "sections_surveillees": list(orchestrator.memory._get_all_sections()),
        "performance_moyenne": f"{stats['performance']['avg_processing_time']:.2f}s",
        "recommandations_systeme": orchestrator._get_system_recommendations()
    }
    
    return report
```

### Intégration avec Systèmes Externes

```python
def integrate_with_security_system(orchestrator):
    """Intégration avec système de sécurité externe"""
    
    result = orchestrator.analyze(video_path, section, time, density)
    
    if result.alert_level == "high":
        # Notification système sécurité
        security_system.trigger_alert(
            level=result.alert_level,
            location=section,
            details=result.recommendation,
            confidence=result.confidence
        )
        
        # Log dans système central
        central_log.record_event({
            "timestamp": result.timestamp,
            "alert_type": result.llm_decision.alert_type,
            "location": section,
            "ai_confidence": result.confidence
        })
```