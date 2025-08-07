# Guide de Déploiement

## Prérequis Système

### Ressources Minimales

**Configuration de Base** :
- CPU : 4 cœurs, 8GB RAM
- Stockage : 20GB SSD libre
- Python : 3.8 ou supérieur
- Système : Linux/Windows/macOS

**Configuration GPU Recommandée** :
- NVIDIA GPU avec 4GB+ VRAM
- CUDA 11.8 ou supérieur
- Driver NVIDIA compatible

### Vérification des Prérequis

```bash
# Vérification Python
python --version  # Doit être >= 3.8

# Vérification CUDA (optionnel)
nvidia-smi

# Vérification espace disque
df -h  # Linux/macOS
dir   # Windows
```

## Installation Production

### 1. Environnement Virtuel

```bash
# Création environnement
python -m venv surveillance_env

# Activation
# Linux/macOS:
source surveillance_env/bin/activate
# Windows:
surveillance_env\Scripts\activate

# Mise à jour pip
pip install --upgrade pip
```

### 2. Installation des Dépendances

```bash
# Installation production
pip install -r requirements.txt

# Vérification de l'installation
python -c "import torch; print('PyTorch OK')"
python -c "from transformers import pipeline; print('Transformers OK')"
```

### 3. Configuration de Production

#### Fichier `.env`
```bash
# Configuration production
SURVEILLANCE_PRIMARY_VLM=smolvlm
SURVEILLANCE_LOG_LEVEL=INFO
SURVEILLANCE_BATCH_SIZE=4
SURVEILLANCE_MAX_GPU_MEMORY=0.8
SURVEILLANCE_CLEANUP_AFTER_ANALYSIS=true

# Chemins de données
SURVEILLANCE_DATA_PATH=/var/lib/surveillance_orchestrator
SURVEILLANCE_LOG_PATH=/var/log/surveillance_orchestrator
```

#### Permissions et Dossiers

```bash
# Création des dossiers nécessaires
sudo mkdir -p /var/lib/surveillance_orchestrator
sudo mkdir -p /var/log/surveillance_orchestrator

# Attribution des permissions
sudo chown $USER:$USER /var/lib/surveillance_orchestrator
sudo chown $USER:$USER /var/log/surveillance_orchestrator
```

## Déploiement Docker

### Dockerfile

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Dépendances système
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Répertoire de travail
WORKDIR /app

# Installation Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code
COPY . .

# Variables d'environnement
ENV SURVEILLANCE_PRIMARY_VLM=smolvlm
ENV SURVEILLANCE_LOG_LEVEL=INFO
ENV SURVEILLANCE_DATA_PATH=/app/data
ENV SURVEILLANCE_LOG_PATH=/app/logs

# Création des dossiers
RUN mkdir -p /app/data /app/logs

# Port d'exposition (si API REST)
EXPOSE 8000

# Commande par défaut
CMD ["python", "main.py", "--help"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  surveillance_orchestrator:
    build: .
    container_name: surveillance_orchestrator
    restart: unless-stopped
    
    # GPU support
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./videos:/app/videos:ro
    
    environment:
      - SURVEILLANCE_PRIMARY_VLM=smolvlm
      - SURVEILLANCE_BATCH_SIZE=4
      - SURVEILLANCE_LOG_LEVEL=INFO
    
    # Healthcheck
    healthcheck:
      test: ["CMD", "python", "-c", "import torch; print('OK')"]
      interval: 30s
      timeout: 10s
      retries: 3
    
    # Limits
    mem_limit: 8g
    cpus: 4.0
```

### Commandes Docker

```bash
# Build de l'image
docker build -t surveillance_orchestrator .

# Lancement avec GPU
docker run --gpus all -v $(pwd)/videos:/app/videos surveillance_orchestrator

# Avec docker-compose
docker-compose up -d
```

## Service Système (Linux)

### Fichier Service

```ini
# /etc/systemd/system/surveillance-orchestrator.service
[Unit]
Description=Surveillance Orchestrator Service
After=network.target
Wants=network.target

[Service]
Type=simple
User=surveillance
Group=surveillance
WorkingDirectory=/opt/surveillance_orchestrator
ExecStart=/opt/surveillance_orchestrator/venv/bin/python main.py --daemon
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10

# Variables d'environnement
Environment=SURVEILLANCE_LOG_LEVEL=INFO
Environment=SURVEILLANCE_DATA_PATH=/var/lib/surveillance_orchestrator

# Limits
LimitNOFILE=4096
MemoryMax=8G

[Install]
WantedBy=multi-user.target
```

### Gestion du Service

```bash
# Installation
sudo cp surveillance-orchestrator.service /etc/systemd/system/
sudo systemctl daemon-reload

# Activation
sudo systemctl enable surveillance-orchestrator
sudo systemctl start surveillance-orchestrator

# Status et logs
sudo systemctl status surveillance-orchestrator
sudo journalctl -u surveillance-orchestrator -f
```

## Monitoring et Supervision

### Monitoring avec Prometheus

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'surveillance_orchestrator'
    static_configs:
      - targets: ['localhost:8080']
    scrape_interval: 30s
    metrics_path: /metrics
```

### Logging Centralisé

```python
# Configuration logging production
import logging
import logging.handlers

def setup_production_logging():
    # Rotation des logs
    handler = logging.handlers.RotatingFileHandler(
        '/var/log/surveillance_orchestrator/app.log',
        maxBytes=50*1024*1024,  # 50MB
        backupCount=5
    )
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
```

## Optimisations de Performance

### Configuration GPU Production

```python
# Optimisations CUDA
import torch

def optimize_gpu_settings():
    if torch.cuda.is_available():
        # Optimisation mémoire
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # Cache management
        torch.cuda.empty_cache()
        
        print(f"GPU optimisé: {torch.cuda.get_device_name(0)}")
```

### Monitoring Ressources

```python
import psutil
import time
from threading import Thread

class ResourceMonitor:
    def __init__(self):
        self.monitoring = True
    
    def start_monitoring(self):
        Thread(target=self._monitor_loop, daemon=True).start()
    
    def _monitor_loop(self):
        while self.monitoring:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            if cpu_percent > 90:
                logger.warning(f"🔥 CPU élevé: {cpu_percent:.1f}%")
            
            if memory_percent > 85:
                logger.warning(f"💾 Mémoire élevée: {memory_percent:.1f}%")
            
            time.sleep(30)
```

## Sécurité et Accès

### Configuration Sécurisée

```bash
# Création utilisateur dédié
sudo useradd -r -s /bin/false -m -d /opt/surveillance_orchestrator surveillance

# Permissions restreintes
sudo chmod 750 /opt/surveillance_orchestrator
sudo chown -R surveillance:surveillance /opt/surveillance_orchestrator
```

### Firewall et Réseau

```bash
# UFW (Ubuntu/Debian)
sudo ufw allow from 192.168.1.0/24 to any port 8080
sudo ufw enable

# Restriction d'accès
# Utiliser un reverse proxy nginx avec authentification
```

## Backup et Récupération

### Script de Sauvegarde

```bash
#!/bin/bash
# backup_surveillance.sh

BACKUP_DIR="/backup/surveillance_orchestrator"
DATE=$(date +%Y%m%d_%H%M%S)

# Création sauvegarde
mkdir -p "$BACKUP_DIR/$DATE"

# Sauvegarde données
cp -r /var/lib/surveillance_orchestrator "$BACKUP_DIR/$DATE/"
cp -r /opt/surveillance_orchestrator/config "$BACKUP_DIR/$DATE/"

# Sauvegarde logs récents (7 derniers jours)
find /var/log/surveillance_orchestrator -name "*.log" -mtime -7 -exec cp {} "$BACKUP_DIR/$DATE/" \;

# Compression
tar -czf "$BACKUP_DIR/surveillance_backup_$DATE.tar.gz" -C "$BACKUP_DIR" "$DATE"
rm -rf "$BACKUP_DIR/$DATE"

# Nettoyage anciennes sauvegardes (> 30 jours)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +30 -delete

echo "✅ Sauvegarde créée: surveillance_backup_$DATE.tar.gz"
```

## Troubleshooting

### Problèmes Communs

#### Erreur CUDA
```bash
# Vérification
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Solution: Réinstallation PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Erreur Mémoire
```bash
# Monitoring mémoire
free -h
ps aux --sort=-%mem | head

# Solution: Ajustement configuration
export SURVEILLANCE_BATCH_SIZE=1
export SURVEILLANCE_CLEANUP_AFTER_ANALYSIS=true
```

#### Permissions
```bash
# Vérification permissions
ls -la /var/lib/surveillance_orchestrator
ls -la /var/log/surveillance_orchestrator

# Correction
sudo chown -R surveillance:surveillance /var/lib/surveillance_orchestrator
sudo chmod -R 755 /var/lib/surveillance_orchestrator
```

### Logs de Debug

```bash
# Activation debug
export SURVEILLANCE_LOG_LEVEL=DEBUG

# Analyse des logs
tail -f /var/log/surveillance_orchestrator/app.log

# Recherche d'erreurs
grep -n "ERROR" /var/log/surveillance_orchestrator/app.log
```