# Exemples de Base

## Analyse Simple

### Exemple 1 : Première Analyse
```bash
# Analyse avec paramètres par défaut
python main.py videos/surveillance_test.mp4
```

**Résultat attendu** :
- Extraction de ~15 frames
- Analyse avec SmolVLM
- Décision contextuelle avec Phi-3
- Rapport détaillé

### Exemple 2 : Paramètres Personnalisés
```bash
python main.py videos/caisse.mp4 \
    --section "Caisse centrale" \
    --time-of-day "Soirée" \
    --crowd-density "faible" \
    --verbose
```

## Utilisation de la Mémoire Contextuelle

### Analyse avec Apprentissage
```python
from src.orchestrator.controller import SurveillanceOrchestrator

orchestrator = SurveillanceOrchestrator()

# Première analyse - apprentissage initial
result1 = orchestrator.analyze("video1.mp4", "Entrée", "Matin", "dense")

# Deuxième analyse - utilise l'apprentissage  
result2 = orchestrator.analyze("video2.mp4", "Entrée", "Matin", "dense")
# Le système reconnaît les patterns et s'améliore

# Insights intelligents
insights = orchestrator.get_section_insights("Entrée")
print(f"Zone analysée {insights['total_analyses']} fois")
print(f"Taux d'alerte: {insights['alert_rate']:.1f}%")
```

## Gestion des Modèles

### Vérification des Ressources
```bash
# État complet du système
python scripts/model_manager.py --diagnostics

# État GPU spécifiquement  
python scripts/model_manager.py --gpu-status
```

### Basculement de Modèles
```bash
# Activer KIM si possible
python scripts/model_manager.py --enable-kim

# Basculer vers KIM
python scripts/model_manager.py --switch-to kim

# Retour vers SmolVLM
python scripts/model_manager.py --switch-to smolvlm
```

## Optimisation des Performances

### Configuration selon GPU
```python
from src.utils.memory_optimizer import memory_optimizer

# Auto-configuration
memory_optimizer.auto_configure_settings()

# Vérification de la pression mémoire
if memory_optimizer.check_memory_pressure():
    memory_optimizer.aggressive_cleanup()
```

### Extraction Intelligente
```bash
# Pour vidéos longues - extraction de keyframes
python main.py long_video.mp4 --keyframes

# Pour ressources limitées - batch réduit
python main.py video.mp4 --batch-size 1
```