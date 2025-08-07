#!/usr/bin/env python3
"""
Script de test et démonstration des optimisations mémoire.
Permet de mesurer l'impact des différentes configurations.
"""

import sys
from pathlib import Path

# Ajout du répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.memory_optimizer import MemoryOptimizer, memory_monitor, optimize_model_loading
from src.config import settings, ModelType
from src.utils.logging import get_surveillance_logger

logger = get_surveillance_logger()


def test_memory_monitoring():
    """Test du monitoring mémoire."""
    print("🧪 TEST MONITORING MÉMOIRE")
    print("=" * 50)
    
    optimizer = MemoryOptimizer()
    
    # État initial
    optimizer.log_memory_status("INITIAL")
    
    # Simulation charge mémoire
    with memory_monitor("Test charge mémoire"):
        # Création d'objets pour tester
        big_list = [i for i in range(1000000)]
        logger.info("Liste créée")
        
        # Test pression mémoire
        pressure = optimizer.check_memory_pressure()
        print(f"Pression mémoire: {'OUI' if pressure else 'NON'}")
        
        del big_list
    
    # Nettoyage
    optimizer.aggressive_cleanup()


def test_auto_optimization():
    """Test de l'auto-optimisation."""
    print("\n🎯 TEST AUTO-OPTIMISATION")
    print("=" * 50)
    
    optimizer = MemoryOptimizer()
    
    # Sauvegarde config actuelle
    original_batch_size = settings.config.batch_size
    original_cleanup = settings.config.cleanup_after_analysis
    original_max_frames = settings.config.processing.max_frames
    
    print("Configuration avant optimisation:")
    print(f"  Batch size: {original_batch_size}")
    print(f"  Cleanup auto: {original_cleanup}")
    print(f"  Max frames: {original_max_frames}")
    
    # Auto-optimisation
    optimizer.auto_configure_settings()
    
    print("\nConfiguration après optimisation:")
    print(f"  Batch size: {settings.config.batch_size}")
    print(f"  Cleanup auto: {settings.config.cleanup_after_analysis}")
    print(f"  Max frames: {settings.config.processing.max_frames}")
    
    # Restauration
    settings.config.batch_size = original_batch_size
    settings.config.cleanup_after_analysis = original_cleanup
    settings.config.processing.max_frames = original_max_frames


def test_model_optimization():
    """Test des optimisations de chargement de modèles."""
    print("\n🤖 TEST OPTIMISATIONS MODÈLES")
    print("=" * 50)
    
    models_to_test = ["smolvlm", "kim", "phi3"]
    
    for model_name in models_to_test:
        print(f"\nOptimisations pour {model_name}:")
        config = optimize_model_loading(model_name)
        
        for key, value in config.items():
            print(f"  {key}: {value}")


def benchmark_memory_usage():
    """Benchmark de l'usage mémoire selon les configurations."""
    print("\n📊 BENCHMARK CONFIGURATIONS MÉMOIRE")
    print("=" * 50)
    
    optimizer = MemoryOptimizer()
    info = optimizer._get_memory_info()
    
    # Test différentes configurations
    configs = [
        {"batch_size": 1, "cleanup": True, "name": "Mode économique"},
        {"batch_size": 2, "cleanup": True, "name": "Mode équilibré"},
        {"batch_size": 4, "cleanup": False, "name": "Mode performance"},
        {"batch_size": 8, "cleanup": False, "name": "Mode gourmand"}
    ]
    
    for config in configs:
        print(f"\n{config['name']}:")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Cleanup: {config['cleanup']}")
        
        # Estimation mémoire
        if "gpu_total_gb" in info:
            estimated_usage = config['batch_size'] * 1.5  # GB par batch
            gpu_percent = (estimated_usage / info['gpu_total_gb']) * 100
            print(f"  Usage GPU estimé: {estimated_usage:.1f}GB ({gpu_percent:.1f}%)")
            
            if gpu_percent > 90:
                print(f"  ⚠️  RISQUE: Usage GPU très élevé!")
            elif gpu_percent > 70:
                print(f"  ⚠️  Attention: Usage GPU élevé")
            else:
                print(f"  ✅ Usage GPU acceptable")


def simulate_analysis_scenarios():
    """Simule différents scénarios d'analyse."""
    print("\n🎬 SIMULATION SCÉNARIOS D'ANALYSE")
    print("=" * 50)
    
    scenarios = [
        {
            "name": "Vidéo courte (30s)",
            "frames": 15,
            "complexity": "faible"
        },
        {
            "name": "Vidéo moyenne (2min)",
            "frames": 60,
            "complexity": "moyenne"
        },
        {
            "name": "Vidéo longue (10min)",
            "frames": 300,
            "complexity": "élevée"
        }
    ]
    
    optimizer = MemoryOptimizer()
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  Frames à traiter: {scenario['frames']}")
        print(f"  Complexité: {scenario['complexity']}")
        
        # Calcul recommandations
        if scenario['frames'] > 100:
            print("  🔧 Recommandations:")
            print("    - Utiliser l'extraction de keyframes")
            print("    - Réduire la taille de batch")
            print("    - Activer le nettoyage automatique")
        elif scenario['frames'] > 50:
            print("  🔧 Recommandations:")
            print("    - Traitement par batch")
            print("    - Surveillance mémoire active")
        else:
            print("  ✅ Configuration standard suffisante")


def memory_tips():
    """Affiche des conseils d'optimisation mémoire."""
    print("\n💡 CONSEILS D'OPTIMISATION MÉMOIRE")
    print("=" * 50)
    
    tips = [
        "🔋 Utilisez cleanup_after_analysis=True pour économiser la mémoire",
        "📏 Réduisez batch_size si vous manquez de VRAM",
        "🎞️ Utilisez l'extraction de keyframes pour de longues vidéos",
        "🧠 SmolVLM consomme moins de mémoire que KIM",
        "🔄 Le basculement automatique évite les surcharges",
        "📊 Surveillez régulièrement l'usage mémoire avec --diagnostics",
        "🧹 Le nettoyage agressif libère la mémoire en cas de besoin",
        "⚡ Désactivez le nettoyage auto pour plus de performances",
        "🎯 L'auto-optimisation adapte les paramètres à votre GPU",
        "📱 load_in_8bit réduit l'usage VRAM de 50% pour KIM"
    ]
    
    for tip in tips:
        print(f"  {tip}")


def main():
    """Fonction principale."""
    print("🧠 TESTEUR D'OPTIMISATIONS MÉMOIRE")
    print("Surveillance Orchestrator")
    print("=" * 60)
    
    try:
        # Tests séquentiels
        test_memory_monitoring()
        test_auto_optimization()
        test_model_optimization()
        benchmark_memory_usage()
        simulate_analysis_scenarios()
        memory_tips()
        
        print("\n" + "=" * 60)
        print("✅ TOUS LES TESTS TERMINÉS")
        print("\nUtilisez ces informations pour optimiser votre configuration:")
        print("  python scripts/model_manager.py --diagnostics")
        print("  python main.py --help")
        
    except Exception as e:
        logger.error(f"Erreur dans les tests: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())