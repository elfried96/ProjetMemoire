# 🛡️ Surveillance Orchestrator

## Vue d'ensemble

**Surveillance Orchestrator** est un système de surveillance intelligente qui utilise des modèles Vision-Language (VLM) et de langage (LLM) pour analyser automatiquement des vidéos de surveillance et détecter des comportements suspects.

![Architecture](assets/architecture-overview.png)

## 🎯 Objectifs

- **Réduction des faux positifs** : Moins de 3% de faux positifs grâce à l'IA avancée
- **Analyse temps réel** : Traitement quasi temps réel de plusieurs flux vidéo
- **Détection intelligente** : Reconnaissance de patterns de comportement suspects
- **Orchestration dynamique** : Coordination automatique d'outils spécialisés

## 🏗️ Architecture

Le système repose sur une architecture modulaire à trois niveaux :

### 1. Modèles d'IA
- **SmolVLM** : Modèle Vision-Language principal (optimisé pour les ressources limitées)
- **KIM** : Modèle VLM avancé (nécessite plus de ressources GPU)
- **Phi-3** : Modèle LLM pour la prise de décision

### 2. Orchestrateur Central
- Coordination des analyses
- Gestion de la mémoire contextuelle  
- Prise de décision intelligente
- Basculement automatique entre modèles

### 3. Outils Spécialisés
- Preprocessing vidéo avancé
- Extraction de keyframes intelligente
- Logging et monitoring
- Interface de configuration

## 🚀 Fonctionnalités

### Analyse Intelligente
- ✅ Détection de vol et dissimulation d'objets
- ✅ Analyse de regroupements inhabituels
- ✅ Identification de gestes suspects
- ✅ Reconnaissance d'objets abandonnés

### Gestion de Modèles
- ✅ Basculement dynamique SmolVLM ↔ KIM
- ✅ Optimisation mémoire GPU/CPU
- ✅ Configuration centralisée
- ✅ Monitoring des performances

### Interface Utilisateur
- ✅ CLI avec paramètres avancés
- ✅ Interface de démonstration interactive
- ✅ Utilitaires de gestion des modèles
- ✅ Logs structurés

## 📊 Exemple d'Utilisation

### Analyse Simple
```bash
python main.py videos/surveillance_test.mp4 --section "Rayon cosmétique"
```

### Analyse Avancée avec Keyframes
```bash
python main.py videos/surveillance_test.mp4 \
    --section "Entrée principale" \
    --time-of-day "Soirée" \
    --crowd-density "dense" \
    --keyframes \
    --model smolvlm
```

### Gestion des Modèles
```bash
# Vérifier l'état des modèles
python scripts/model_manager.py --status

# Activer KIM si les ressources le permettent
python scripts/model_manager.py --enable-kim

# Basculer vers KIM
python scripts/model_manager.py --switch-to kim
```

## 🎬 Démonstration

Lancez la démonstration interactive :

```bash
python demo.py
```

La démo offre :
- Analyse interactive de vidéos
- Test de tous les modèles disponibles
- Basculement en temps réel entre modèles
- Statistiques détaillées

## 📈 Résultats Typiques

Le système analyse une vidéo et retourne :

```json
{
  "suspicion_level": "medium",
  "alert_type": "dissimulation",
  "reasoning": "Personne observée cachant un objet dans sa veste...",
  "action": "surveiller_discretement",
  "confidence": 0.87,
  "recommended_tools": ["detection_objets", "analyse_mouvement"]
}
```

## 🔧 Configuration Flexible

Le système s'adapte automatiquement aux ressources disponibles :

- **CPU uniquement** : SmolVLM en mode économique
- **GPU < 6GB** : SmolVLM avec optimisations
- **GPU ≥ 8GB** : KIM avec capacités avancées

## 📚 Documentation

- [Installation](installation.md) - Guide d'installation complet
- [Guide Utilisateur](user-guide/quickstart.md) - Premiers pas
- [Architecture](architecture/overview.md) - Détails techniques
- [API](api/config.md) - Documentation de l'API
- [Exemples](examples/basic.md) - Cas d'usage pratiques

## 🤝 Contribution

Ce projet est développé dans le cadre d'un projet de licence en Intelligence Artificielle à l'IFRI.

**Auteur** : Elfried Steve David Kinzoun  
**Année** : 2025  
**Université** : IFRI (Institut de Formation et de Recherche en Informatique)

## 📄 Licence

Projet académique — tous droits réservés © 2025  
Usage limité à des fins de démonstration, d'expérimentation et de recherche.