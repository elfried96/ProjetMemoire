# Surveillance Orchestrator - Explication Complète du Code

## Vue d'Ensemble du Projet

Le **Surveillance Orchestrator** est un système de surveillance intelligente qui utilise l'intelligence artificielle pour analyser des vidéos de surveillance et détecter automatiquement des comportements suspects dans un environnement commercial (magasins). Le système combine des modèles VLM (Vision-Language Models) pour l'analyse visuelle et des modèles LLM (Large Language Models) pour la prise de décision contextuelle.

## Architecture Générale

```
Vidéo d'entrée → Extraction de frames → Analyse VLM → Décision LLM → Alerte/Action
                        ↓                    ↓            ↓
                  Préprocessing      Mémoire & Patterns  Logging & Stats
```

## Structure des Fichiers

### 1. Point d'Entrée Principal

#### `main.py` (lines 1-251)
**Rôle :** Point d'entrée du système avec interface en ligne de commande complète.

**Fonctionnalités principales :**
- **Configuration CLI** (lines 17-83) : Gestion des arguments (vidéo, section, modèle, etc.)
- **Gestion des modèles** (lines 86-102) : Affichage des modèles VLM disponibles
- **Orchestration** (lines 155-248) : Lance l'analyse complète d'une vidéo
- **Affichage des résultats** (lines 104-153) : Formatage intelligent des résultats avec niveaux d'alerte

**Points clés :**
- Supporte deux modèles VLM : SmolVLM (par défaut) et KIM
- Modes d'extraction : frames régulières ou keyframes intelligentes
- Gestion robuste des erreurs et interruptions utilisateur

### 2. Orchestrateur Principal

#### `src/orchestrator/controller.py` (lines 1-473)
**Rôle :** Cerveau du système qui coordonne tous les composants.

**Classes principales :**

##### `MemoryManager` (lines 22-119)
- Gestion de l'historique des événements et alertes
- Sauvegarde JSON avec nettoyage automatique (max 1000 événements)
- Récupération d'alertes récentes par section

##### `SurveillanceOrchestrator` (lines 121-473)
Orchestrateur principal avec méthodes clés :
- **`analyze()`** (lines 160-224) : Pipeline complet d'analyse
- **`_extract_frames()`** (lines 226-237) : Extraction de frames vidéo
- **`_analyze_with_vlm()`** (lines 239-280) : Analyse avec modèle vision
- **`_make_decision()`** (lines 282-324) : Prise de décision avec LLM
- **`_build_final_result()`** (lines 326-353) : Construction du résultat final

**Fonctionnalités avancées :**
- Basculement automatique entre modèles VLM
- Gestion mémoire intelligente avec nettoyage
- Statistiques de session en temps réel
- Support de l'analyse par batch pour grandes vidéos

### 3. Configuration Système

#### `src/config/settings.py` (lines 1-160)
**Rôle :** Configuration centralisée et gestion des modèles.

**Classes de configuration :**
- **`ModelType`** (lines 13-16) : Énumération des modèles VLM
- **`ModelConfig`** (lines 19-27) : Configuration d'un modèle (ID, device, etc.)
- **`ProcessingConfig`** (lines 31-38) : Paramètres de traitement vidéo
- **`SurveillanceConfig`** (lines 42-76) : Configuration globale du système

**Classe `Settings`** (lines 93-160) - Singleton de gestion :
- Configuration des modèles disponibles
- Basculement dynamique entre VLM
- Chargement depuis variables d'environnement
- Création automatique des répertoires de travail

### 4. Modèles d'IA

#### `src/models/base.py` (lines 1-231)
**Rôle :** Classes abstraites et interfaces communes pour tous les modèles.

**Classes de données :**
- **`AnalysisResult`** (lines 14-24) : Résultat d'analyse VLM
- **`SuspicionAnalysis`** (lines 28-39) : Résultat d'analyse de suspicion LLM

**Classes abstraites :**
- **`BaseVLMModel`** (lines 42-144) : Interface pour modèles vision
  - Méthodes obligatoires : `load_model()`, `unload_model()`, `analyze_images()`
  - Méthode spécialisée : `analyze_surveillance_scene()` avec prompt personnalisé
- **`BaseLLMModel`** (lines 146-188) : Interface pour modèles de langage
  - Méthode principale : `analyze_context()` pour prendre des décisions

**`ModelManager`** (lines 190-231) : Gestionnaire de cycle de vie des modèles
- Enregistrement et récupération de modèles
- Chargement automatique à la demande
- Nettoyage mémoire intelligent

#### `src/models/smolvlm_wrapper.py` (lines 1-287)
**Rôle :** Wrapper pour le modèle SmolVLM (modèle VLM principal).

**`SmolVLMWrapper`** - Implémentation complète :
- **Chargement** (lines 36-64) : Utilise AutoProcessor et AutoModelForCausalLM
- **Analyse d'images** (lines 88-164) : Pipeline complet avec gestion d'erreurs
- **Extraction thinking/summary** (lines 166-189) : Parse les sections de réflexion du modèle
- **Estimation de confiance** (lines 191-214) : Score basé sur des mots-clés
- **Analyse par batch** (lines 216-283) : Traitement de grandes séquences vidéo

**Fonctionnalités techniques :**
- Support GPU avec fallback CPU
- Gestion mémoire avec torch.no_grad()
- Prompt spécialisé pour surveillance avec sections thinking

#### `src/models/phi3_wrapper.py` (lines 1-312)
**Rôle :** Wrapper pour Phi-3 (modèle LLM de prise de décision).

**`Phi3Wrapper`** - Fonctionnalités :
- **Construction de prompt** (lines 87-140) : Prompt structuré avec contexte complet
- **Analyse contextuelle** (lines 142-197) : Prise de décision basée sur analyse VLM
- **Extraction JSON** (lines 199-262) : Parse et valide la réponse structurée
- **Analyse de fallback** (lines 268-308) : Analyse basée sur mots-clés si JSON échoue

**Critères de décision :**
- Niveaux de suspicion : low/medium/high
- Types d'alertes : observation, dissimulation, repérage, tentative_vol, etc.
- Actions recommandées : surveiller, alerter_agent, intervenir, etc.

### 5. Préprocessing Vidéo

#### `src/utils/preprocessing.py` (lines 1-367)
**Rôle :** Extraction et amélioration d'images à partir de vidéos.

**`VideoProcessor`** - Classe principale :
- **Informations vidéo** (lines 38-84) : Analyse complète (fps, durée, résolution)
- **Extraction de frames** (lines 86-153) : Frames régulières avec préprocessing
- **Préprocessing d'images** (lines 155-211) : Redimensionnement, contraste, débruitage
- **Redimensionnement intelligent** (lines 213-251) : Préserve le ratio d'aspect
- **Extraction de keyframes** (lines 253-324) : Détection de changements significatifs

**Algorithmes d'amélioration :**
- Redimensionnement adaptatif avec padding
- Amélioration de contraste et luminosité
- Filtre médian pour réduction de bruit
- Détection de keyframes par différence d'images

### 6. Mémoire Intelligente

#### `src/orchestrator/memory_engine.py` (lines 1-393)
**Rôle :** Système d'apprentissage et de mémoire contextuelle avancé.

**Classes intelligentes :**

##### `PatternDetector` (lines 18-67)
- Apprentissage de patterns comportementaux suspects vs normaux
- Calcul de similarité entre contextes
- Reconnaissance de situations similaires

##### `ContextualLearning` (lines 70-151)
- Statistiques par section (taux d'alertes, confiance moyenne)
- Suivi des performances des modèles
- Recommandations de modèles selon le contexte

##### `IntelligentMemoryManager` (lines 153-393)
**Fonctionnalités avancées :**
- Cache en mémoire avec expiration (5 minutes)
- Sauvegarde intelligente avec apprentissage automatique
- Alertes contextuelles basées sur l'historique
- Optimisation automatique de l'usage mémoire

**Apprentissage automatique :**
- Détection de patterns récurrents
- Adaptation des recommandations de modèles
- Insights statistiques par section surveillée

### 7. Utilitaires

#### `src/utils/logging.py` (lines 1-61)
**Rôle :** Système de logging unifié pour tout le projet.

**Fonctions :**
- **`setup_logger()`** (lines 7-52) : Configuration flexible de logger
- **`get_surveillance_logger()`** (lines 55-61) : Logger principal du projet

**Fonctionnalités :**
- Sortie console + fichier optionnel
- Format standardisé avec timestamps
- Niveaux configurables (INFO par défaut)

#### `src/utils/memory_optimizer.py` et `src/utils/io.py`
Ces fichiers contiennent des utilitaires pour l'optimisation mémoire et les opérations d'E/S, mentionnés dans le code principal mais non détaillés dans cette analyse.

## Flux de Données Principal

### 1. Initialisation
```
Settings → Configuration modèles → Création orchestrateur → Chargement mémoire
```

### 2. Analyse d'une vidéo
```
Vidéo → VideoProcessor → Frames → SmolVLM → Résultat VLM
                                      ↓
Contexte + Historique ← IntelligentMemoryManager
                                      ↓
                              Phi3 → Décision → Sauvegarde
```

### 3. Apprentissage continu
```
Chaque analyse → PatternDetector + ContextualLearning → Mémoire améliorée
```

## Points Forts du Système

### 1. **Modularité et Extensibilité**
- Interfaces abstraites permettent l'ajout facile de nouveaux modèles
- Configuration centralisée pour adaptation rapide
- Composants découplés et réutilisables

### 2. **Gestion Mémoire Intelligente**
- Chargement/déchargement automatique des modèles
- Cache avec expiration pour optimiser les performances
- Nettoyage automatique des données anciennes

### 3. **Robustesse**
- Gestion d'erreurs à tous les niveaux
- Mécanismes de fallback (analyse par mots-clés si JSON échoue)
- Validation des données avec valeurs par défaut

### 4. **Apprentissage Adaptatif**
- Détection automatique de patterns suspects
- Recommandations de modèles basées sur les performances passées
- Insights statistiques par zone surveillée

### 5. **Scalabilité**
- Support de l'analyse par batch pour grandes vidéos
- Optimisations mémoire pour fonctionnement continu
- Configuration flexible selon les ressources disponibles

## Cas d'Usage

Le système est conçu pour la surveillance de magasins avec capacités de :
- **Détection de vol** : Dissimulation d'objets, mouvements suspects vers sorties
- **Analyse comportementale** : Repérage, nervosité, patterns inhabituels
- **Surveillance adaptative** : Apprentissage des zones à risque
- **Alertes intelligentes** : Recommandations d'actions selon le niveau de suspicion

## Sécurité et Éthique

Le code respecte les bonnes pratiques de sécurité :
- Aucune exposition de secrets ou clés
- Usage défensif uniquement (détection, pas d'action malveillante)
- Logging transparent des décisions prises
- Respect de la vie privée (pas de reconnaissance faciale intrusive)

Ce système représente une approche moderne et intelligente de la surveillance automatisée, combinant vision par ordinateur, traitement du langage naturel et apprentissage adaptatif pour une sécurité proactive et contextuelle.