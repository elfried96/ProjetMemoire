# Utilisation Avancée

## Mémoire Contextuelle Intelligente

Le système dispose d'une **mémoire contextuelle avancée** qui apprend des analyses précédentes pour améliorer les performances.

### Fonctionnalités d'Apprentissage

#### 1. Détection de Patterns
```python
from src.orchestrator.controller import SurveillanceOrchestrator

orchestrator = SurveillanceOrchestrator()

# L'orchestrateur apprend automatiquement des patterns
result = orchestrator.analyze(video_path, section, time_of_day, crowd_density)

# Récupère les insights d'une section
insights = orchestrator.get_section_insights("Rayon cosmétique")
print(f"Taux d'alerte: {insights['alert_rate']:.1f}%")
```

#### 2. Optimisation Automatique
Le système s'adapte automatiquement selon l'historique :

- **Alertes contextuelles** : Prend en compte les patterns similaires
- **Recommandations de modèles** : Suggère le meilleur modèle selon le contexte
- **Apprentissage des sections** : Mémorise les spécificités de chaque zone

### Configuration Avancée des Modèles

#### Basculement Intelligent
```python
# Le système recommande automatiquement le meilleur modèle
orchestrator = SurveillanceOrchestrator()

# Basculement manuel si nécessaire
if orchestrator.switch_to_kim():
    print("✅ Passé à KIM")
else:
    print("⚠️ KIM non disponible, reste sur SmolVLM")
```

#### Optimisation Mémoire
```python
# Optimisation automatique selon les ressources
from src.utils.memory_optimizer import memory_optimizer

memory_optimizer.auto_configure_settings()

# Nettoyage intelligent de la mémoire contextuelle
orchestrator.optimize_contextual_memory()
```

### Analyse par Keyframes Intelligente

```bash
# Extraction intelligente pour longues vidéos
python main.py videos/long_video.mp4 --keyframes --verbose

# Avec paramètres optimisés pour performance
python main.py videos/video.mp4 \
    --keyframes \
    --batch-size 2 \
    --model smolvlm \
    --section "Zone sensible"
```

### Monitoring des Performances

#### Statistiques Avancées
```python
stats = orchestrator.get_session_stats()
print(f"Intelligence: {stats['intelligence']['total_patterns']} patterns appris")
print(f"Sections analysées: {stats['intelligence']['sections_learned']}")
```

#### Insights par Section
```python
insights = orchestrator.get_section_insights("Rayon électronique")
if insights.get("alert_rate", 0) > 30:
    print("⚠️ Zone à haut risque détectée")
```

## Intégration avec d'Autres Systèmes

### API REST (Future)
```python
# Sera disponible prochainement
from src.api.server import SurveillanceAPI

api = SurveillanceAPI()
api.start_server(port=8080)
```

### Webhooks (Future)
```python
# Configuration webhooks pour alertes
settings.config.webhooks = {
    "high_alert": "https://your-system.com/alert",
    "medium_alert": "https://monitoring.com/warn"
}
```