# Architecture du Système de Surveillance Multimodal

## Chapitre 2 : Méthodologie et Architecture du Système Multimodal

### 2.1 Approche méthodologique

#### 2.1.1 Méthodologie de développement
- **Approche itérative** : Développement par cycles courts avec validation continue
- **Architecture modulaire** : Séparation claire des responsabilités (VLM, LLM, orchestrateur)
- **Évaluation continue** : Tests automatisés et métriques de performance en temps réel
- **Apprentissage contextuel** : Système adaptatif qui améliore ses performances avec l'usage

#### 2.1.2 Paradigme multimodal
- **Vision-Language Models (VLM)** : Analyse des images de surveillance
- **Large Language Models (LLM)** : Prise de décision contextuelle et raisonnement
- **Fusion intelligente** : Combinaison des résultats pour décisions optimales
- **Mémoire contextuelle** : Apprentissage des patterns et amélioration continue

#### 2.1.3 Méthodologie d'évaluation
- **Métriques quantitatives** : Précision, rappel, F1-score, temps de traitement
- **Métriques qualitatives** : Cohérence des décisions, explicabilité du raisonnement
- **Tests de stress** : Performance sous charge, gestion mémoire, robustesse
- **Validation terrain** : Tests avec données réelles de surveillance

### 2.2 Architecture de la solution proposée

#### 2.2.1 Architecture globale
```
┌─────────────────────────────────────────────────────────────┐
│                    SURVEILLANCE ORCHESTRATOR                │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   INPUT     │  │ PROCESSING  │  │       OUTPUT        │  │
│  │   LAYER     │  │   LAYER     │  │       LAYER         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

#### 2.2.2 Couches architecturales

**Couche d'Entrée (Input Layer)**
- **Gestionnaire de vidéos** : Extraction et préprocessing des frames
- **Extraction intelligente** : Keyframes basée sur détection de changements
- **Optimisation mémoire** : Gestion adaptative selon ressources disponibles

**Couche de Traitement (Processing Layer)**
- **Module VLM** : Analyse visuelle avec SmolVLM/KIM
- **Module LLM** : Raisonnement contextuel avec Phi-3
- **Moteur de décision** : Fusion des analyses et prise de décision finale
- **Mémoire contextuelle** : Apprentissage et adaptation continue

**Couche de Sortie (Output Layer)**
- **Système d'alertes** : Classification par niveaux de suspicion
- **Reporting détaillé** : Logs structurés et explicabilité
- **Interface utilisateur** : Démonstration interactive et monitoring

#### 2.2.3 Composants principaux

```
src/
├── orchestrator/          # Contrôleur principal
│   ├── controller.py      # Orchestration centrale
│   └── memory_engine.py   # Moteur d'apprentissage
├── models/               # Wrappers des modèles IA
│   ├── base.py          # Classes de base abstraites
│   ├── smolvlm_wrapper.py # Interface SmolVLM
│   ├── phi3_wrapper.py    # Interface Phi-3
│   └── kim_wrapper.py     # Interface KIM (fallback)
├── utils/               # Utilitaires système
│   ├── preprocessing.py  # Traitement vidéo/images
│   ├── memory_optimizer.py # Optimisation mémoire
│   └── logging.py       # Système de logs avancé
└── config/              # Configuration système
    └── settings.py      # Paramètres globaux
```

### 2.3 Choix technologiques

#### 2.3.1 Modèles d'IA sélectionnés

**Vision-Language Model (VLM) Principal : SmolVLM**
- **Avantages** : Optimisé pour analyse multi-images, faible empreinte mémoire
- **Performance** : ~1.7B paramètres, inference rapide sur GPU consumer
- **Spécialisation** : Excellent pour détection d'objets et comportements

**Large Language Model (LLM) : Phi-3**
- **Avantages** : Raisonnement avancé, excellent rapport qualité/taille
- **Performance** : ~2.7B paramètres, temps de réponse <1s
- **Spécialisation** : Analyse contextuelle et prise de décision structurée

#### 2.3.2 Stack technologique

**Core Framework**
- **Python 3.12+** : Langage principal pour IA/ML
- **PyTorch 2.7+** : Framework de deep learning
- **Transformers 4.53+** : Interfaces modèles Hugging Face

**Traitement Multimédia**
- **OpenCV 4.12+** : Traitement vidéo et extraction de frames
- **Pillow 11.3+** : Manipulation d'images
- **FFmpeg** : Décodage vidéo optimisé

**Optimisation Performance**
- **Accelerate 1.9+** : Optimisation GPU multi-device
- **BitsAndBytes 0.46+** : Quantization pour efficacité mémoire
- **TorchVision 0.22+** : Preprocessing optimisé

### 2.4 Spécifications du prototype

#### 2.4.1 Spécifications fonctionnelles

**Capacités d'analyse**
- **Formats supportés** : MP4, AVI, MOV, MKV
- **Résolutions** : 480p à 4K (optimisation automatique)
- **Durée vidéo** : Jusqu'à 10 minutes par analyse
- **Framerate** : Adaptation automatique selon contenu

**Types de détection**
- **Comportements suspects** : Vol, vandalisme, intrusion
- **Analyses contextuelles** : Affluence, heure, zone
- **Niveaux d'alerte** : Low, Medium, High avec actions associées
- **Confiance** : Score de 0.0 à 1.0 avec seuils adaptatifs

#### 2.4.2 Spécifications techniques

**Ressources système**
- **RAM minimum** : 8GB (16GB recommandé)
- **GPU** : CUDA compatible, 6GB+ VRAM
- **Stockage** : 50GB pour modèles + données
- **CPU** : 4+ cœurs, support AVX2/AVX512

**Performance cible**
- **Latence analyse** : <60s pour vidéo 5min
- **Throughput** : 10+ vidéos/heure
- **Précision** : >85% détection comportements suspects
- **Disponibilité** : 99.5% uptime en production

## Chapitre 3 : Implémentation et Évaluation Expérimentale

### 3.1 Implémentation du système

#### 3.1.1 Architecture de classes

```python
# Hiérarchie des modèles
BaseVLMModel (ABC)
├── SmolVLMWrapper    # Modèle principal
├── KIMWrapper        # Fallback/alternative
└── CustomVLMWrapper  # Extensions futures

BaseLLMModel (ABC)
├── Phi3Wrapper       # Modèle de raisonnement
└── CustomLLMWrapper  # Extensions futures

# Orchestration
SurveillanceOrchestrator
├── ModelManager      # Gestion des modèles
├── MemoryEngine      # Apprentissage contextuel
└── PerformanceMonitor # Métriques temps réel
```

#### 3.1.2 Pipeline de traitement

1. **Preprocessing** : Extraction frames + optimisation
2. **Analyse VLM** : Description visuelle détaillée
3. **Contextualisation** : Enrichissement avec mémoire
4. **Décision LLM** : Raisonnement et classification
5. **Post-processing** : Formatage et sauvegarde

#### 3.1.3 Gestion mémoire adaptative

```python
# Auto-optimisation selon GPU disponible
def optimize_for_gpu(gpu_memory_gb):
    if gpu_memory_gb >= 16:
        return {"batch_size": 8, "max_frames": 20}
    elif gpu_memory_gb >= 8:
        return {"batch_size": 4, "max_frames": 15}
    else:
        return {"batch_size": 2, "max_frames": 10}
```

### 3.2 Protocoles d'évaluation

#### 3.2.1 Datasets d'évaluation

**Dataset principal** : Vidéos surveillance retail (2h+ contenu)
- **Normal** : 70% - Comportements clients standards
- **Suspect** : 20% - Comportements potentiellement suspects
- **Critique** : 10% - Incidents avérés (vol, vandalisme)

**Métriques d'évaluation**
- **Précision** : VP/(VP+FP) par classe de suspicion
- **Rappel** : VP/(VP+FN) pour détection incidents
- **F1-Score** : Moyenne harmonique précision/rappel
- **Latence** : Temps moyen traitement par vidéo

#### 3.2.2 Protocole de test

1. **Tests unitaires** : Chaque composant isolément
2. **Tests d'intégration** : Pipeline complet
3. **Tests de performance** : Charge et stress
4. **Tests de régression** : Validation après modifications
5. **Tests terrain** : Validation environnement réel

---

## Architecture Multi-Caméras : Évolution vers un Système Distribué

### Approche pour Architecture Multi-Caméras

#### Phase 1 : Extension Actuelle (Immédiat)

**Modification du Controller**
```python
class MultiCameraSurveillanceOrchestrator:
    def __init__(self):
        self.camera_analyzers = {}  # Un orchestrateur par caméra
        self.fusion_engine = CameraFusionEngine()
        self.global_memory = GlobalMemoryEngine()
    
    def analyze_multi_camera(self, camera_streams: Dict[str, Path]):
        results = {}
        for camera_id, video_path in camera_streams.items():
            results[camera_id] = self.analyze_single_camera(camera_id, video_path)
        
        return self.fusion_engine.fuse_results(results)
```

**Gestionnaire de Fusion**
```python
class CameraFusionEngine:
    def fuse_results(self, camera_results: Dict[str, dict]) -> dict:
        # Corrélation temporelle entre caméras
        # Tracking d'objets inter-caméras
        # Décision globale multi-perspectives
        pass
```

#### Phase 2 : Architecture Distribuée (Moyen terme)

**Composants additionnels nécessaires :**

1. **Camera Manager**
   ```python
   class CameraManager:
       def __init__(self):
           self.active_cameras = {}
           self.camera_configs = {}
           self.sync_manager = TemporalSyncManager()
   ```

2. **Temporal Synchronization**
   ```python
   class TemporalSyncManager:
       def synchronize_streams(self, streams: Dict[str, Any]):
           # Synchronisation temporelle multi-flux
           # Compensation latence réseau
           # Buffer management
   ```

3. **Cross-Camera Tracking**
   ```python
   class PersonTracker:
       def track_across_cameras(self, detections: Dict[str, List]):
           # Réidentification inter-caméras
           # Suivi de trajectoires globales
           # Détection comportements multi-zones
   ```

#### Phase 3 : Système Temps Réel (Long terme)

**Architecture Microservices**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Camera Node 1  │    │  Camera Node 2  │    │  Camera Node N  │
│  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │
│  │ VLM Local │  │    │  │ VLM Local │  │    │  │ VLM Local │  │
│  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │   CENTRAL FUSION NODE    │
                    │  ┌─────────────────────┐  │
                    │  │  Global LLM Brain   │  │
                    │  │  Decision Engine    │  │
                    │  │  Memory Engine      │  │
                    │  └─────────────────────┘  │
                    └───────────────────────────┘
```

### Étapes de Migration Multi-Caméras

#### Étape 1 : Préparation de l'Architecture
```bash
# 1. Créer les nouveaux modules
mkdir -p src/multi_camera/{fusion,tracking,sync}

# 2. Étendre la configuration
# config/multi_camera_settings.py

# 3. Adapter la base de données
# Ajouter support multiple streams temporels
```

#### Étape 2 : Implémentation Core
```python
# src/multi_camera/fusion/camera_fusion.py
class CameraFusionEngine:
    def __init__(self):
        self.camera_results = {}
        self.temporal_window = 5.0  # secondes
        self.confidence_threshold = 0.7
    
    def add_camera_result(self, camera_id: str, timestamp: float, result: dict):
        """Ajoute résultat d'une caméra avec timestamp"""
        
    def get_fused_decision(self, timestamp: float) -> dict:
        """Décision globale basée sur toutes les caméras actives"""
        
    def detect_cross_camera_events(self) -> List[dict]:
        """Détecte événements impliquant plusieurs caméras"""
```

#### Étape 3 : Tracking Inter-Caméras
```python
# src/multi_camera/tracking/person_tracker.py
class CrossCameraPersonTracker:
    def __init__(self):
        self.active_tracks = {}
        self.reid_model = PersonReIdentificationModel()
        
    def update_tracks(self, camera_id: str, detections: List[dict]):
        """Met à jour les tracks avec nouvelles détections"""
        
    def get_person_trajectory(self, person_id: str) -> List[dict]:
        """Récupère trajectoire complète d'une personne"""
```

#### Étape 4 : Configuration Multi-Caméras
```python
# config/multi_camera_config.py
CAMERA_LAYOUT = {
    "entrance": {
        "position": (0, 0),
        "coverage_angle": 120,
        "priority": "high",
        "overlap_cameras": ["lobby"]
    },
    "cashier": {
        "position": (10, 5),
        "coverage_angle": 90,
        "priority": "critical",
        "overlap_cameras": ["entrance", "exit"]
    },
    "aisles": {
        "position": (5, 10),
        "coverage_angle": 180,
        "priority": "medium",
        "overlap_cameras": ["cashier"]
    }
}
```

### Avantages Architecture Multi-Caméras

#### Avantages Techniques
- **Couverture complète** : Élimination des angles morts
- **Redondance** : Fiabilité accrue par validation croisée
- **Tracking continu** : Suivi personnes sur tout le magasin
- **Détection avancée** : Patterns comportementaux complexes

#### Avantages Métier
- **Réduction false positives** : Validation multi-perspectives
- **Détection précoce** : Patterns suspects dès l'entrée
- **Evidence robuste** : Preuves multi-angles pour incidents
- **ROI amélioré** : Optimisation staffing sécurité

### Défis et Solutions

#### Défis Techniques
1. **Synchronisation temporelle** → NTP sync + buffer management
2. **Scalabilité** → Architecture distribuée + load balancing
3. **Corrélation spatiale** → Mapping 3D + calibration caméras
4. **Performance réseau** → Compression adaptative + edge computing

#### Solutions Proposées
1. **Edge Computing** : Traitement local par nœud caméra
2. **Streaming optimisé** : Compression intelligente selon contenu
3. **Decision aggregation** : Fusion bayésienne des confidences
4. **Fallback graceful** : Dégradation en mode single-camera

### Roadmap d'Implémentation

#### Phase 1 (1-2 mois) : Foundation
- [ ] Refactoring architecture actuelle
- [ ] Implémentation CameraManager
- [ ] Tests multi-flux basiques

#### Phase 2 (2-3 mois) : Fusion Engine
- [ ] Algorithmes de fusion de décisions
- [ ] Corrélation temporelle
- [ ] Interface configuration multi-caméras

#### Phase 3 (3-4 mois) : Advanced Features
- [ ] Tracking inter-caméras
- [ ] Réidentification de personnes
- [ ] Détection comportements complexes

#### Phase 4 (4-6 mois) : Production Ready
- [ ] Optimisation performance temps réel
- [ ] Interface monitoring avancée
- [ ] Déploiement production

### Métriques de Succès Multi-Caméras

#### KPIs Techniques
- **Latence end-to-end** : <2s pour décision globale
- **Synchronisation** : <100ms drift entre caméras
- **Disponibilité** : >99.9% uptime système global
- **Scalabilité** : Support 10+ caméras simultanées

#### KPIs Métier
- **Précision détection** : >90% avec validation multi-angles
- **Réduction false alerts** : -50% vs single camera
- **Couverture incidents** : >95% événements détectés
- **ROI sécurité** : Optimisation staffing 20-30%

Cette architecture évolutive permet une migration progressive vers un système multi-caméras enterprise-grade tout en conservant la robustesse et les performances du système actuel.