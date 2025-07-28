# 🛡️ Surveillance Orchestrator – VLM-Driven Smart Monitoring System

> **Projet de Licence – Intelligence Artificielle**  
> _Par Elfried Steve David Kinzoun_  
> Année : 2025  
> Université : IFRI

---

## 📍 Contexte

Les grandes surfaces commerciales font face à des défis sécuritaires complexes, notamment en matière de prévention des vols. Les systèmes de vidéosurveillance traditionnels génèrent un volume massif de **faux positifs**, ce qui surcharge les équipes humaines et réduit l'efficacité opérationnelle.

Le projet **Surveillance Orchestrator** propose une **solution intelligente basée sur un modèle Vision-Language (KIM)** capable d'**orchestrer dynamiquement des outils spécialisés** comme YOLO, DeepSORT, etc., pour réduire drastiquement les faux positifs et détecter efficacement les comportements suspects.

---

## 🧠 Objectifs du projet

- 🎯 Détecter automatiquement les comportements suspects dans des vidéos de surveillance
- 🧠 Utiliser un **modèle VLM (KIM)** comme cerveau décisionnel
- 🧰 Orchestration dynamique d'outils spécialisés : YOLOv8, DeepSORT, validateurs contextuels
- 🛡️ Réduction des faux positifs à moins de 3%
- ⚡ Traitement **quasi temps réel** de plusieurs flux vidéo

---

## 🧱 Architecture du projet

```bash
surveillance_orchestrator/
├── pyproject.toml              # Gestion des dépendances via UV
├── README.md                   # Ce fichier
├── src/
│   ├── core/                   # Orchestrateur principal
│   ├── models/                 # KIM + moteur de prompt
│   ├── tools/                  # Outils de surveillance (YOLO, tracker, etc.)
│   ├── utils/                  # Aide, logs, conversions
│   └── __init__.py
├── videos/                     # Vidéos d'entrée pour test
├── data/                       # Images, metadata
├── outputs/                    # Frames suspectes, logs, rapports
├── notebooks/                  # Analyses exploratoires ou tests
└── .venv/                      # Environnement virtuel UV
```

---

## ⚙️ Technologies utilisées

| Catégorie        | Outils / Frameworks                           |
| ---------------- | --------------------------------------------- |
| 🧠 Modèle VLM    | [`KIM`](https://huggingface.co/MBZUAI/KIM-v1) |
| 🎥 Vision        | `OpenCV`, `PIL`, `YOLOv8` via `ultralytics`   |
| 🔎 Orchestration | Prompts personnalisés + logique métier Python |
| 🚀 Infra         | `uv`, `PyTorch`, `Transformers`, `CUDA`       |
| 🐳 Déploiement   | Docker (prévu), Redis (cache)                 |
| 🛠️ Langage      | Python 3.10+                                  |

---

## 🚀 Lancement rapide

### 1. Installer `uv` (si ce n'est pas déjà fait)

```bash
curl -Ls https://astral.sh/uv/install.sh | bash
```

### 2. Cloner le repo

```bash
git clone https://github.com/ton-profil/surveillance_orchestrator.git
cd surveillance_orchestrator
```

### 3. Initialiser le projet

```bash
uv venv
uv pip install -r requirements.txt
```

### 4. Tester un premier traitement vidéo

```bash
python src/main.py
```

---

## 📦 Dépendances principales

Tu peux les installer avec `uv` :

```bash
uv  torch torchvision
uv  transformers
uv  opencv-python pillow
uv  ultralytics
```

---

## 🎯 Fonctionnement général

1. 🖼️ Lecture d'une vidéo
2. 📤 Extraction de frames clés (ex: toutes les 5 sec)
3. 💬 Génération d'un **prompt intelligent** (contexte, heure, zone, etc.)
4. 🧠 Envoi du prompt + image à **KIM**
5. ⚙️ Si comportement suspect : appel à **YOLOv8** pour confirmation
6. 📁 Export des alertes : logs, frames, rapports

---

## 🧪 Scénarios pris en charge

* Détection de **vol ou dissimulation d'objets**
* Analyse de **regroupement inhabituel**
* Identification de **gestes suspects vers les sorties**
* Reconnaissance d'**objets abandonnés**

---

## 📈 Améliorations futures

* [ ] Interface web en Streamlit ou FastAPI
* [ ] Multi-caméras + synchronisation
* [ ] Traitement parallèle GPU (multiprocessing ou Ray)
* [ ] Analyse audio/vidéo combinée (multimodalité étendue)
* [ ] Génération automatique de rapports PDF

---

## 🧾 Licence

Projet académique — tous droits réservés © 2025
Usage limité à des fins de démonstration, d'expérimentation et de recherche.

---

## 📬 Contact

> **Elfried Kinzoun**
> Licence en Intelligence Artificielle
> ✉️ kinzoun.elfried[at]exemple.com
> 📍 Bénin