# 📥 Information sur le Téléchargement des Modèles

## Comment ça fonctionne

### 🔄 Téléchargement Automatique
Lors du **premier lancement**, les modèles sont téléchargés automatiquement via Hugging Face :

1. **SmolVLM** (`HuggingFaceTB/SmolVLM-Instruct`) - ~1.5 GB
2. **Phi-3** (`microsoft/phi-2`) - ~2.7 GB
3. **KIM** (`microsoft/kosmos-2-patch14-224`) - ~1.6 GB (optionnel)

### 📂 Stockage Local
Les modèles sont mis en cache dans :
```
~/.cache/huggingface/hub/
├── models--HuggingFaceTB--SmolVLM-Instruct/
├── models--microsoft--phi-2/
└── models--microsoft--kosmos-2-patch14-224/
```

### ⏱️ Temps de Téléchargement (estimé)
- **Connexion rapide (100 Mbps)** : 5-10 minutes
- **Connexion normale (20 Mbps)** : 15-30 minutes  
- **Connexion lente (5 Mbps)** : 45-90 minutes

### 🚀 Optimisations dans le Code

#### Chargement Différé (par défaut)
```python
# Les modèles ne se chargent QUE quand nécessaires
if settings.config.cleanup_after_analysis:
    logger.info("Modèle initialisé (chargement différé)")
else:
    self.load_model()  # Chargement immédiat
```

#### Nettoyage Automatique
```python
# Libère la mémoire après chaque analyse
if settings.config.cleanup_after_analysis:
    self._cleanup_models()
```

## 🛠️ Contrôle Manuel du Téléchargement

### Pré-télécharger les modèles
```python
from transformers import AutoModelForCausalLM, AutoProcessor

# Force le téléchargement de SmolVLM
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")

# Force le téléchargement de Phi-3
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
```

### Variables d'Environnement
```bash
# Cache personnalisé
export TRANSFORMERS_CACHE="/mon/cache/custom"

# Offline mode (utilise seulement les modèles déjà téléchargés)
export TRANSFORMERS_OFFLINE=1
```

## 🎯 Premier Lancement Recommandé

### Étape 1 : Installation
```bash
pip install torch torchvision transformers accelerate
```

### Étape 2 : Test de téléchargement
```bash
# Lance le téléchargement + test
python main.py --list-models
```

### Étape 3 : Première analyse
```bash
# Utilise SmolVLM (plus léger)
python main.py videos/surveillance_test.mp4 --verbose
```

## 🔧 Gestion des Erreurs de Téléchargement

### Erreur réseau
```python
# Le code gère automatiquement les reprises
try:
    model = AutoModelForCausalLM.from_pretrained(model_name)
except Exception as e:
    logger.error(f"Erreur téléchargement: {e}")
    # Fallback ou nouvelle tentative
```

### Espace disque insuffisant
- **SmolVLM + Phi-3** : ~4.5 GB requis minimum
- **Avec KIM** : ~6 GB total

### Mode Offline
Si déjà téléchargé, fonctionne sans connexion :
```bash
export TRANSFORMERS_OFFLINE=1
python main.py videos/test.mp4
```

## 📊 Monitoring du Téléchargement

Le projet affiche la progression :
```
🔄 Téléchargement de SmolVLM: HuggingFaceTB/SmolVLM-Instruct
Downloading... ████████████████████████ 100% 1.2GB/1.2GB
✅ SmolVLM chargé avec succès sur cuda
```

## 🎛️ Configuration Avancée

### Chargement en 8-bit (économise la mémoire)
```python
# Automatique pour KIM
load_in_8bit=True if self.device == "cuda" else False
```

### Device mapping automatique
```python
# Répartit automatiquement sur GPU disponible
device_map="auto" if self.device == "cuda" else None
```