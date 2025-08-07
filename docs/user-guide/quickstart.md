# Démarrage Rapide

Ce guide vous permet de commencer à utiliser Surveillance Orchestrator en moins de 5 minutes.

## 🎯 Objectif

Analyser votre première vidéo de surveillance et comprendre le fonctionnement du système.

## Étape 1 : Vérification

Assurez-vous que l'installation est correcte :

```bash
python scripts/model_manager.py --status
```

Vous devriez voir quelque chose comme :

```
🎯 CONFIGURATION ACTUELLE
==================================================
✅   🎯 SmolVLM (PRINCIPAL)
      Status: ✅ ACTIVÉ
      Disponible: ✅ DISPONIBLE
      ID Modèle: HuggingFaceTB/SmolVLM-Instruct

❌     KIM
      Status: ❌ DÉSACTIVÉ
      Disponible: ❌ NON DISPONIBLE
      ID Modèle: microsoft/kosmos-2-patch14-224
```

## Étape 2 : Préparation d'une Vidéo

### Option A : Utiliser les Vidéos de Démo

Le projet inclut des vidéos de test dans le dossier `videos/` :

```bash
ls videos/
```

### Option B : Ajouter Votre Vidéo

Copiez votre vidéo dans le dossier `videos/` :

```bash
cp /chemin/vers/votre/video.mp4 videos/ma_video.mp4
```

!!! tip "Formats Supportés"
    - MP4, AVI, MOV, MKV, FLV, WMV
    - Résolution : 720p à 4K
    - Durée recommandée : 30s à 5min pour les tests

## Étape 3 : Première Analyse

### Analyse Simple

```bash
python main.py videos/surveillance_test.mp4
```

Cette commande utilise les paramètres par défaut :
- Section : "Rayon cosmétique" 
- Moment : "Fin d'après-midi"
- Affluence : "dense"

### Analyse avec Paramètres

```bash
python main.py videos/surveillance_test.mp4 \
    --section "Entrée principale" \
    --time-of-day "Soirée" \
    --crowd-density "faible"
```

## Étape 4 : Comprendre le Résultat

L'analyse retourne un rapport structuré :

```
🧠 RÉSULTAT DE L'ANALYSE DE SURVEILLANCE
================================================================================
📍 Section: Rayon cosmétique
🕐 Timestamp: 2025-01-06T10:30:45
⚡ Durée: 15.7s
🤖 Modèle utilisé: SmolVLM

🧠 RÉFLEXION DU VLM:
----------------------------------------
Je vois plusieurs personnes dans le rayon. Une personne près des étagères 
semble manipuler des produits de façon inhabituelle...

📊 DESCRIPTION DE LA SCÈNE:
----------------------------------------
Scène de magasin avec 3-4 personnes visibles. Éclairage normal.
Une personne effectue des gestes suspects près des produits cosmétiques.

⚖️  DÉCISION DE L'ORCHESTRATEUR:
----------------------------------------
Niveau de suspicion: MEDIUM
Type d'alerte: dissimulation
Action recommandée: surveiller_discretement
Confiance: 0.75

💭 RAISONNEMENT:
Comportement de dissimulation détecté avec niveau de confiance moyen.
Recommande une surveillance discrète sans intervention directe.

⚠️  Surveillance renforcée recommandée
================================================================================
```

## Étape 5 : Exploration des Options

### Voir Toutes les Options

```bash
python main.py --help
```

### Options Utiles

```bash
# Mode verbeux pour plus de détails
python main.py videos/test.mp4 --verbose

# Extraction intelligente de keyframes  
python main.py videos/test.mp4 --keyframes

# Forcer un modèle spécifique
python main.py videos/test.mp4 --model smolvlm

# Ajuster la taille de lot
python main.py videos/test.mp4 --batch-size 2
```

## Étape 6 : Démonstration Interactive

Lancez l'interface de démonstration :

```bash
python demo.py
```

La démo offre un menu interactif :

```
📋 MENU PRINCIPAL
------------------------------
1. Analyser une vidéo
2. Analyser toutes les vidéos de démo  
3. Changer de modèle VLM
4. Afficher les statistiques
5. Test des capacités système
6. Quitter
```

!!! tip "Conseils pour la Démo"
    - Commencez par l'option 5 pour tester votre système
    - Utilisez l'option 1 pour des analyses détaillées
    - L'option 2 est parfaite pour tester différents scénarios

## Étape 7 : Gestion des Modèles

### Vérifier l'État GPU

```bash
python scripts/model_manager.py --gpu-status
```

### Activer KIM (si GPU ≥ 8GB)

```bash
python scripts/model_manager.py --enable-kim
python scripts/model_manager.py --switch-to kim
```

### Diagnostic Complet

```bash
python scripts/model_manager.py --diagnostics
```

## Cas d'Usage Typiques

### 1. Surveillance d'Entrée

```bash
python main.py videos/entree.mp4 \
    --section "Entrée principale" \
    --time-of-day "Matin" \
    --crowd-density "dense"
```

### 2. Surveillance de Caisse

```bash  
python main.py videos/caisse.mp4 \
    --section "Caisse" \
    --time-of-day "Après-midi" \
    --crowd-density "modérée" \
    --keyframes
```

### 3. Surveillance Nocturne

```bash
python main.py videos/nuit.mp4 \
    --section "Rayon électronique" \
    --time-of-day "Soirée" \
    --crowd-density "faible" \
    --verbose
```

## Interprétation des Résultats

### Niveaux de Suspicion

| Niveau | Description | Action Typique |
|--------|-------------|----------------|
| **LOW** | Situation normale | Aucune action |
| **MEDIUM** | Surveillance nécessaire | Observation discrète |
| **HIGH** | Intervention requise | Alerte sécurité |

### Types d'Alertes

| Type | Signification |
|------|---------------|
| `rien` | Aucun comportement suspect |
| `observation` | Comportement à surveiller |
| `dissimulation` | Tentative de cacher des objets |
| `repérage` | Surveillance des lieux |
| `tentative_vol` | Vol en cours |
| `comportement_suspect` | Autre comportement anormal |

### Actions Recommandées

| Action | Description |
|--------|-------------|
| `rien` | Aucune intervention |
| `surveiller_discretement` | Observer sans révéler |
| `alerter_agent` | Prévenir la sécurité |
| `intervenir` | Intervention directe |
| `demander_renfort` | Appeler des renforts |

## Résolution de Problèmes Courants

### Vidéo Non Trouvée

```bash
❌ Erreur: La vidéo 'test.mp4' n'existe pas.
```

**Solution** : Vérifiez le chemin et utilisez un chemin absolu si nécessaire :
```bash
python main.py /chemin/complet/vers/video.mp4
```

### Analyse Lente

Si l'analyse prend trop de temps :

```bash
# Réduisez la taille de lot
python main.py videos/test.mp4 --batch-size 1

# Utilisez les keyframes pour moins d'images
python main.py videos/test.mp4 --keyframes
```

### Mémoire Insuffisante

```bash
# Vérifiez l'état de votre système
python scripts/model_manager.py --diagnostics

# Forcez l'utilisation CPU
export CUDA_VISIBLE_DEVICES=""
python main.py videos/test.mp4
```

## Prochaines Étapes

Maintenant que vous maîtrisez les bases :

1. 📖 Consultez le [Guide Avancé](advanced.md) pour des fonctionnalités plus poussées
2. ⚙️ Lisez la [Configuration](configuration.md) pour personnaliser le système
3. 🏗️ Explorez l'[Architecture](../architecture/overview.md) pour comprendre le fonctionnement interne
4. 💡 Découvrez plus d'[Exemples](../examples/basic.md) d'utilisation

!!! success "Félicitations !"
    Vous savez maintenant utiliser Surveillance Orchestrator ! 🎉