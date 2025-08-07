# FAQ (Questions Fréquentes)

## Installation et Configuration

### Q: Quels sont les prérequis minimum pour faire fonctionner le système ?

**A:** Les prérequis minimum sont :
- Python 3.8 ou supérieur
- 8GB de RAM 
- 20GB d'espace disque libre
- CPU 4 cœurs minimum

Pour de meilleures performances, un GPU NVIDIA avec 4GB+ VRAM est recommandé.

### Q: Comment savoir si mon GPU est compatible ?

**A:** Vérifiez avec ces commandes :

```bash
# Vérification NVIDIA
nvidia-smi

# Test PyTorch CUDA
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"

# Mémoire GPU
python -c "import torch; print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB')"
```

Le système fonctionne également sans GPU, mais les performances seront réduites.

### Q: Quelle est la différence entre SmolVLM et KIM ?

**A:** 

| Modèle | VRAM Requise | Performance | Précision | Usage Recommandé |
|--------|-------------|-------------|-----------|------------------|
| SmolVLM | ~2-4GB | Rapide | Bonne | Ressources limitées, surveillance générale |
| KIM | ~6-8GB | Lent | Excellente | GPU puissant, analyses critiques |

SmolVLM est le modèle par défaut, KIM est utilisé quand les ressources le permettent.

### Q: Comment changer les paramètres de configuration ?

**A:** Trois méthodes :

1. **Fichier `.env`** :
```bash
SURVEILLANCE_PRIMARY_VLM=kim
SURVEILLANCE_BATCH_SIZE=2
```

2. **Configuration Python** :
```python
from src.config import settings, ModelType
settings.config.primary_vlm = ModelType.KIM
settings.config.batch_size = 2
```

3. **Variables d'environnement** :
```bash
export SURVEILLANCE_BATCH_SIZE=2
python main.py video.mp4
```

## Utilisation et Fonctionnalités

### Q: Comment analyser plusieurs vidéos en même temps ?

**A:** Utilisez un script Python avec threading :

```python
from concurrent.futures import ThreadPoolExecutor
from src.orchestrator.controller import SurveillanceOrchestrator

orchestrator = SurveillanceOrchestrator()

def analyze_video(video_path):
    return orchestrator.analyze(video_path, "Zone A", "Après-midi", "dense")

videos = ["video1.mp4", "video2.mp4", "video3.mp4"]

with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(analyze_video, videos))
```

### Q: Comment le système apprend-il des patterns ?

**A:** Le système utilise un moteur de mémoire contextuelle qui :

1. **Enregistre automatiquement** chaque analyse
2. **Détecte des patterns** récurrents par section/heure
3. **Apprend des corrélations** entre contexte et niveau de suspicion
4. **S'améliore progressivement** avec plus d'analyses

Exemple de pattern appris automatiquement :
```json
{
  "section": "Caisse 1",
  "time": "Après-midi",
  "density": "dense",
  "typical_suspicion": "low",
  "confidence": 0.87,
  "observations": 25
}
```

### Q: Comment interpréter les niveaux d'alerte ?

**A:** Le système utilise trois niveaux :

- **Low** : Situation normale, aucune action requise
- **Medium** : Situation à surveiller, vigilance recommandée  
- **High** : Situation suspecte, action immédiate recommandée

La **confidence** (0.0-1.0) indique la fiabilité de l'analyse.

### Q: Que faire si le système détecte beaucoup de fausses alertes ?

**A:** Plusieurs solutions :

1. **Ajuster les seuils** :
```python
# Réduction de sensibilité
settings.config.processing.quality_threshold = 0.9
```

2. **Laisser le système apprendre** (il s'améliore avec le temps)

3. **Analyser les patterns** :
```python
insights = orchestrator.get_section_insights("Zone problématique")
print(f"Taux d'alerte: {insights['alert_rate']:.1f}%")
```

4. **Optimiser la configuration** selon la zone surveillée

## Performance et Optimisation

### Q: Le système est lent, comment l'accélérer ?

**A:** Plusieurs optimisations possibles :

1. **Réduire le batch size** :
```bash
export SURVEILLANCE_BATCH_SIZE=1
```

2. **Utiliser les keyframes** :
```bash
python main.py video.mp4 --keyframes
```

3. **Activer le nettoyage automatique** :
```bash
export SURVEILLANCE_CLEANUP_AFTER_ANALYSIS=true
```

4. **Vérifier la mémoire GPU** :
```python
python scripts/model_manager.py --gpu-status
```

### Q: Comment surveiller l'utilisation des ressources ?

**A:** Le système inclut un monitoring automatique :

```python
from src.utils.memory_optimizer import memory_monitor

# Monitoring automatique
with memory_monitor("Mon analyse"):
    result = orchestrator.analyze(...)
    
# Stats manuelles
from src.utils.memory_optimizer import memory_optimizer
stats = memory_optimizer.get_memory_stats()
print(f"CPU: {stats['cpu_used_gb']:.1f}GB")
```

### Q: Erreur "CUDA out of memory", que faire ?

**A:** Solutions par ordre de priorité :

1. **Réduire batch_size** :
```python
settings.config.batch_size = 1
```

2. **Activer nettoyage agressif** :
```python
settings.config.cleanup_after_analysis = True
```

3. **Réduire frames analysées** :
```python
settings.config.processing.max_frames = 5
```

4. **Passer à SmolVLM** :
```python
settings.config.primary_vlm = ModelType.SMOLVLM
```

## Intégration et API

### Q: Comment intégrer le système dans une application existante ?

**A:** Utilisez l'API Python :

```python
from src.orchestrator.controller import SurveillanceOrchestrator

class MonSystemeExistant:
    def __init__(self):
        self.orchestrator = SurveillanceOrchestrator()
    
    def analyser_video_surveillance(self, video_path, zone):
        result = self.orchestrator.analyze(
            video_path, zone, "temps_reel", "variable"
        )
        
        if result.alert_level == "high":
            self.envoyer_alerte(result)
        
        return result
```

### Q: Comment récupérer les statistiques d'intelligence ?

**A:** Plusieurs niveaux de statistiques :

```python
# Stats globales de session
session_stats = orchestrator.get_session_stats()

# Stats par section
section_insights = orchestrator.get_section_insights("Caisse 1")

# Stats de mémoire contextuelle
memory_stats = orchestrator.memory.get_learning_stats()
```

### Q: Comment sauvegarder/restaurer la mémoire du système ?

**A:** La mémoire est automatiquement persistée dans `data/memory.json`. Pour la gestion manuelle :

```python
# Sauvegarde manuelle
orchestrator.memory.save_memory()

# Optimisation/nettoyage
stats = orchestrator.optimize_contextual_memory()
print(f"Patterns nettoyés: {stats['patterns_cleaned']}")

# Sauvegarde de l'état complet
import json
with open('backup_memory.json', 'w') as f:
    json.dump(orchestrator.memory.get_full_state(), f)
```

## Dépannage

### Q: Le système ne démarre pas, que vérifier ?

**A:** Checklist de dépannage :

1. **Version Python** :
```bash
python --version  # Doit être >= 3.8
```

2. **Dépendances** :
```bash
pip list | grep torch
pip list | grep transformers
```

3. **Permissions** :
```bash
ls -la data/
ls -la logs/
```

4. **Variables d'environnement** :
```bash
echo $SURVEILLANCE_PRIMARY_VLM
```

### Q: Erreur "Model not found" ou "Failed to load model" ?

**A:** Solutions :

1. **Vérifier la connexion internet** (téléchargement initial des modèles)

2. **Espace disque** :
```bash
df -h  # Vérifier l'espace libre
```

3. **Cache Hugging Face** :
```bash
# Nettoyer le cache si corrompu
rm -rf ~/.cache/huggingface/transformers
```

4. **Forcer le re-téléchargement** :
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("HuggingFaceTB/SmolVLM-Instruct", force_download=True)
```

### Q: Les analyses semblent incohérentes ou bizarres ?

**A:** Causes possibles :

1. **Qualité vidéo** : Vérifier résolution, éclairage, format
2. **Mémoire insuffisante** : Modèle tronqué par manque de ressources
3. **Configuration inadaptée** : Ajuster les paramètres pour le contexte

```python
# Diagnostic de qualité
frames = extract_frames_from_video("problem_video.mp4")
for i, frame in enumerate(frames):
    print(f"Frame {i}: {frame.size}, mode: {frame.mode}")
```

### Q: Comment activer les logs détaillés pour le debug ?

**A:** 

```bash
# Variable d'environnement
export SURVEILLANCE_LOG_LEVEL=DEBUG

# Ou dans le code
import logging
logging.getLogger().setLevel(logging.DEBUG)

# Logs spécifiques à un module
logging.getLogger('src.models').setLevel(logging.DEBUG)
```

### Q: Le système utilise-t-il internet ?

**A:** Oui, pour :

1. **Premier téléchargement** des modèles (une seule fois)
2. **Mise à jour des modèles** (optionnel)

Une fois téléchargés, les modèles fonctionnent entièrement hors ligne. Pour forcer le mode hors ligne :

```python
from transformers import AutoModel
# Mode offline
model = AutoModel.from_pretrained("HuggingFaceTB/SmolVLM-Instruct", local_files_only=True)
```