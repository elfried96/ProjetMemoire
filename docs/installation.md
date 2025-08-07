# Installation

## Prérequis Système

### Matériel Recommandé

=== "Configuration Minimale"
    - **CPU** : Intel i5 / AMD Ryzen 5 ou équivalent
    - **RAM** : 8 GB minimum
    - **Stockage** : 10 GB libres
    - **GPU** : Optionnel (CPU uniquement supporté)

=== "Configuration Recommandée"
    - **CPU** : Intel i7 / AMD Ryzen 7 ou équivalent  
    - **RAM** : 16 GB ou plus
    - **Stockage** : 20 GB libres (SSD recommandé)
    - **GPU** : NVIDIA GTX 1660 / RTX 2060 ou équivalent (6+ GB VRAM)

=== "Configuration Optimale"
    - **CPU** : Intel i9 / AMD Ryzen 9 ou équivalent
    - **RAM** : 32 GB ou plus
    - **Stockage** : 50 GB libres (NVMe SSD)
    - **GPU** : NVIDIA RTX 3070 / RTX 4060 ou équivalent (8+ GB VRAM)

### Logiciels Requis

- **Python 3.10+** (testé avec Python 3.10-3.12)
- **CUDA 11.8+** (pour utilisation GPU, optionnel)
- **Git** (pour cloner le repository)

## Installation

### 1. Clonage du Repository

```bash
git clone https://github.com/ton-profil/surveillance_orchestrator.git
cd surveillance_orchestrator
```

### 2. Installation de UV (Gestionnaire de Paquets)

=== "Linux/macOS"
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"
    ```powershell
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

=== "Via pip"
    ```bash
    pip install uv
    ```

### 3. Création de l'Environnement Virtuel

```bash
uv venv
source .venv/bin/activate  # Linux/macOS
# ou
.venv\Scripts\activate     # Windows
```

### 4. Installation des Dépendances

=== "Installation Standard"
    ```bash
    uv pip install -e .
    ```

=== "Installation avec GPU (CUDA)"
    ```bash
    # Installation PyTorch avec CUDA
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    
    # Installation des autres dépendances
    uv pip install -e .
    ```

=== "Installation Développement"
    ```bash
    uv pip install -e ".[dev]"
    ```

## Vérification de l'Installation

### Test Rapide

```bash
python -c "from src.config import settings; print('✅ Configuration OK')"
```

### Test Complet

```bash
python scripts/model_manager.py --diagnostics
```

Ce script vérifie :
- ✅ Installation Python et dépendances
- ✅ Disponibilité GPU/CUDA
- ✅ État des modèles
- ✅ Modules internes

### Test avec Vidéo

```bash
# Vérifier qu'une vidéo de test existe
ls videos/

# Lancer l'analyse (sans modèles chargés)
python main.py --list-models
```

## Configuration Initiale

### 1. Variables d'Environnement (Optionnel)

Créez un fichier `.env` :

```bash
# Modèle VLM principal
SURVEILLANCE_PRIMARY_VLM=smolvlm

# Niveau de log (DEBUG, INFO, WARNING, ERROR)
SURVEILLANCE_LOG_LEVEL=INFO

# Taille de lot pour le traitement
SURVEILLANCE_BATCH_SIZE=4

# Fraction de VRAM utilisable (0.0-1.0)
SURVEILLANCE_MAX_GPU_MEMORY=0.8
```

### 2. Première Configuration

```bash
# Afficher la configuration actuelle
python scripts/model_manager.py --status

# Activer KIM si vous avez suffisamment de VRAM
python scripts/model_manager.py --enable-kim

# Lancer la démonstration interactive
python demo.py
```

## Dépendances Détaillées

### Dépendances Principales

| Package | Version | Description |
|---------|---------|-------------|
| torch | ≥2.7.1 | Framework ML principal |
| transformers | ≥4.53.3 | Modèles Hugging Face |
| opencv-python | ≥4.12.0 | Traitement vidéo |
| pillow | ≥11.3.0 | Traitement d'images |
| ultralytics | ≥8.3.169 | YOLOv8 (futur) |

### Dépendances de Développement

```bash
# Installation complète développement
uv pip install -e ".[dev,docs,test]"
```

## Résolution de Problèmes

### Erreur CUDA

```bash
❌ CUDA non disponible
```

**Solution** :
1. Vérifiez l'installation NVIDIA CUDA Toolkit
2. Réinstallez PyTorch avec support CUDA :
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Erreur Mémoire GPU

```bash
❌ CUDA out of memory
```

**Solution** :
1. Réduisez la taille de lot :
```bash
export SURVEILLANCE_BATCH_SIZE=2
```
2. Ou utilisez le mode CPU uniquement :
```bash
export SURVEILLANCE_PRIMARY_VLM=smolvlm
```

### Erreur Import

```bash
❌ ModuleNotFoundError: No module named 'src'
```

**Solution** :
```bash
# Assurez-vous d'être dans le bon répertoire
cd surveillance_orchestrator

# Vérifiez l'environnement virtuel
source .venv/bin/activate

# Réinstallez en mode développement
uv pip install -e .
```

### Modèles Non Disponibles

```bash
❌ KIM non disponible - ressources insuffisantes
```

**Solution** :
- KIM nécessite au moins 6 GB VRAM
- Utilisez SmolVLM pour les configurations plus modestes
- Vérifiez la mémoire GPU :
```bash
python scripts/model_manager.py --gpu-status
```

## Performance et Optimisation

### Pour CPU Uniquement
- Utilisez exclusivement SmolVLM
- Réduisez la taille de lot à 1-2
- Activez le nettoyage automatique

### Pour GPU Limité (< 6GB)
- Gardez SmolVLM comme principal
- Batch size : 2-4
- Activez le nettoyage automatique

### Pour GPU Puissant (≥ 8GB)
- Activez et utilisez KIM
- Batch size : 4-8
- Désactivez le nettoyage pour les performances

### Exemple Configuration Optimisée

```python
# Dans un script personnalisé
from src.config import settings, ModelType

# Configuration pour GPU puissant
settings.config.batch_size = 6
settings.config.cleanup_after_analysis = False
settings.set_primary_vlm(ModelType.KIM)
```

## Installation Docker (Futur)

!!! note "À venir"
    L'installation Docker sera disponible dans une version future pour faciliter le déploiement.

```bash
# Sera disponible prochainement
docker build -t surveillance-orchestrator .
docker run -v $(pwd)/videos:/app/videos surveillance-orchestrator
```