#!/usr/bin/env python3
"""
Utilitaire de gestion des modèles pour Surveillance Orchestrator.
Permet d'activer/désactiver KIM et de basculer entre les modèles facilement.
"""

import sys
import argparse
from pathlib import Path

# Ajout du répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings, ModelType
from src.models import get_available_models
from src.models.kim_wrapper import KIMWrapper
from src.utils.logging import get_surveillance_logger
from src.utils.memory_optimizer import MemoryOptimizer

logger = get_surveillance_logger()


def display_current_config():
    """Affiche la configuration actuelle."""
    print("🎯 CONFIGURATION ACTUELLE")
    print("=" * 50)
    
    models = get_available_models()
    
    for model_id, info in models.items():
        status = "✅ ACTIVÉ" if info["enabled"] else "❌ DÉSACTIVÉ"
        available = "✅ DISPONIBLE" if info["available"] else "❌ NON DISPONIBLE"
        primary = " (PRINCIPAL)" if info["is_primary"] else ""
        
        print(f"{info['name']}{primary}")
        print(f"  Status: {status}")
        print(f"  Disponible: {available}")
        print(f"  ID Modèle: {info['model_id']}")
        print()
    
    print(f"Configuration:")
    print(f"  VLM Principal: {settings.config.primary_vlm.value}")
    print(f"  Batch Size: {settings.config.batch_size}")
    print(f"  Nettoyage auto: {'Oui' if settings.config.cleanup_after_analysis else 'Non'}")


def check_gpu_status():
    """Vérifie l'état du GPU."""
    print("🖥️  ÉTAT DU GPU")
    print("=" * 30)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            gpu_memory = torch.cuda.get_device_properties(current_device).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)
            
            print(f"✅ CUDA disponible")
            print(f"Nombre de GPU: {gpu_count}")
            print(f"GPU actuel: {gpu_name}")
            print(f"Mémoire GPU: {gpu_memory_gb:.1f} GB")
            
            # Test d'allocation mémoire
            try:
                test_tensor = torch.randn(1000, 1000).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                print("✅ Test GPU: OK")
            except Exception as e:
                print(f"⚠️  Test GPU: {e}")
        else:
            print("❌ CUDA non disponible")
            print("Le système utilisera le CPU (très lent pour les modèles)")
    
    except ImportError:
        print("❌ PyTorch non installé")


def enable_kim():
    """Active KIM si possible."""
    print("🚀 ACTIVATION DE KIM")
    print("=" * 30)
    
    if not KIMWrapper.is_available():
        print("❌ KIM ne peut pas être activé:")
        
        try:
            import torch
            if not torch.cuda.is_available():
                print("  • CUDA non disponible")
            else:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_memory < 6:
                    print(f"  • GPU insuffisant ({gpu_memory:.1f}GB < 6GB requis)")
        except ImportError:
            print("  • PyTorch non installé")
        
        return False
    
    settings.enable_model(ModelType.KIM, True)
    print("✅ KIM activé avec succès")
    return True


def disable_kim():
    """Désactive KIM."""
    print("⏹️  DÉSACTIVATION DE KIM")
    print("=" * 30)
    
    settings.enable_model(ModelType.KIM, False)
    
    # Bascule vers SmolVLM si KIM était principal
    if settings.config.primary_vlm == ModelType.KIM:
        settings.set_primary_vlm(ModelType.SMOLVLM)
        print("📱 Basculement automatique vers SmolVLM")
    
    print("✅ KIM désactivé")


def switch_primary_model(model_name: str):
    """Bascule le modèle principal."""
    print(f"🔄 BASCULEMENT VERS {model_name.upper()}")
    print("=" * 40)
    
    if model_name.lower() == "smolvlm":
        settings.set_primary_vlm(ModelType.SMOLVLM)
        print("✅ SmolVLM défini comme modèle principal")
        
    elif model_name.lower() == "kim":
        if not settings.get_model_config(ModelType.KIM).enabled:
            print("❌ KIM n'est pas activé")
            print("💡 Utilisez 'python model_manager.py --enable-kim' d'abord")
            return False
            
        if not KIMWrapper.is_available():
            print("❌ KIM n'est pas disponible (ressources insuffisantes)")
            return False
        
        settings.set_primary_vlm(ModelType.KIM)
        print("✅ KIM défini comme modèle principal")
        print("⚠️  Attention: KIM nécessite plus de ressources GPU")
        
    else:
        print(f"❌ Modèle '{model_name}' non reconnu")
        return False
    
    return True


def run_diagnostics():
    """Exécute un diagnostic complet du système."""
    print("🩺 DIAGNOSTIC DU SYSTÈME")
    print("=" * 40)
    
    # Vérification Python
    print(f"Python: {sys.version}")
    
    # Vérification des dépendances critiques
    dependencies = [
        ("torch", "PyTorch pour les modèles ML"),
        ("transformers", "Transformers pour les modèles Hugging Face"),
        ("opencv-python", "OpenCV pour le traitement vidéo"),
        ("pillow", "PIL pour le traitement d'images"),
        ("ultralytics", "YOLOv8 pour la détection d'objets")
    ]
    
    print("\n📦 DÉPENDANCES:")
    for dep, desc in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}: OK - {desc}")
        except ImportError:
            print(f"❌ {dep}: MANQUANT - {desc}")
    
    # État du GPU
    print()
    check_gpu_status()
    
    # Optimisations mémoire
    print("\n🧠 OPTIMISATIONS MÉMOIRE:")
    optimizer = MemoryOptimizer()
    optimizer.log_memory_status("DIAGNOSTIC")
    optimizer.auto_configure_settings()
    
    # État des modèles
    print()
    display_current_config()
    
    # Test d'import des modules
    print("\n🧩 MODULES INTERNES:")
    try:
        from src.config import settings
        print("✅ Configuration: OK")
        
        from src.models import create_vlm_model, create_llm_model
        print("✅ Modèles: OK")
        
        from src.utils.preprocessing import video_processor
        print("✅ Preprocessing: OK")
        
        from src.orchestrator.controller import SurveillanceOrchestrator
        print("✅ Orchestrateur: OK")
        
        from src.utils.memory_optimizer import memory_optimizer
        print("✅ Optimiseur mémoire: OK")
        
    except Exception as e:
        print(f"❌ Erreur import: {e}")
        
    # Recommandations
    print("\n💡 RECOMMANDATIONS:")
    info = optimizer._get_memory_info()
    if "gpu_total_gb" in info:
        gpu_gb = info["gpu_total_gb"]
        if gpu_gb < 4:
            print("  🔋 GPU limité: Utilisez SmolVLM uniquement")
            print("  🔧 batch_size=1, cleanup_after_analysis=True")
        elif gpu_gb < 8:
            print("  ⚖️ GPU moyen: SmolVLM recommandé, KIM possible")
            print("  🔧 batch_size=2, load_in_8bit=True pour KIM")
        else:
            print("  🚀 GPU puissant: KIM disponible avec toutes options")
            print("  🔧 batch_size=4+, performances optimales")
    else:
        print("  💻 Mode CPU: SmolVLM uniquement, patience requise")
        print("  🔧 batch_size=1, max_frames=5")


def setup_argument_parser():
    """Configure les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Gestionnaire de modèles pour Surveillance Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Affiche l'état actuel des modèles"
    )
    
    parser.add_argument(
        "--enable-kim",
        action="store_true",
        help="Active KIM si les ressources le permettent"
    )
    
    parser.add_argument(
        "--disable-kim",
        action="store_true",
        help="Désactive KIM"
    )
    
    parser.add_argument(
        "--switch-to",
        type=str,
        choices=["smolvlm", "kim"],
        help="Bascule vers le modèle spécifié"
    )
    
    parser.add_argument(
        "--gpu-status",
        action="store_true",
        help="Vérifie l'état du GPU"
    )
    
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Exécute un diagnostic complet du système"
    )
    
    parser.add_argument(
        "--memory-test",
        action="store_true",
        help="Lance les tests d'optimisation mémoire"
    )
    
    return parser


def main():
    """Fonction principale."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Si aucun argument, affiche le statut
    if not any(vars(args).values()):
        display_current_config()
        return 0
    
    if args.diagnostics:
        run_diagnostics()
        return 0
    
    if args.memory_test:
        # Import et lancement des tests mémoire
        import subprocess
        memory_test_script = Path(__file__).parent / "memory_test.py"
        result = subprocess.run([sys.executable, str(memory_test_script)])
        return result.returncode
    
    if args.gpu_status:
        check_gpu_status()
        return 0
    
    if args.status:
        display_current_config()
        return 0
    
    if args.enable_kim:
        success = enable_kim()
        if not success:
            return 1
    
    if args.disable_kim:
        disable_kim()
    
    if args.switch_to:
        success = switch_primary_model(args.switch_to)
        if not success:
            return 1
    
    # Affiche l'état final si des changements ont été faits
    if args.enable_kim or args.disable_kim or args.switch_to:
        print("\n" + "="*50)
        display_current_config()
    
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n❌ Interrompu par l'utilisateur")
        exit(130)
    except Exception as e:
        logger.error(f"Erreur inattendue: {e}")
        exit(1)