#!/usr/bin/env python3
"""
Point d'entrée principal du Surveillance Orchestrator.
Système de surveillance intelligente utilisant des modèles VLM et LLM.
"""

import sys
import argparse
from pathlib import Path

from src.orchestrator.controller import SurveillanceOrchestrator
from src.config import settings
from src.utils.logging import get_surveillance_logger
from src.models import get_available_models


def setup_argument_parser() -> argparse.ArgumentParser:
    """Configure l'analyseur d'arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Surveillance Orchestrator - Analyse intelligente de vidéos de surveillance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "video_path", 
        type=str,
        help="Chemin vers la vidéo à analyser"
    )
    
    parser.add_argument(
        "--section", 
        type=str, 
        default="Rayon cosmétique",
        help="Section du magasin surveillée"
    )
    
    parser.add_argument(
        "--time-of-day", 
        type=str, 
        default="Fin d'après-midi",
        help="Moment de la journée (ex: Matin, Après-midi, Soirée)"
    )
    
    parser.add_argument(
        "--crowd-density", 
        type=str, 
        default="dense",
        choices=["faible", "modérée", "dense"],
        help="Densité de la foule"
    )
    
    parser.add_argument(
        "--keyframes", 
        action="store_true",
        help="Utiliser l'extraction de keyframes intelligente"
    )
    
    parser.add_argument(
        "--model", 
        type=str,
        choices=["smolvlm", "kim"],
        help="Force l'utilisation d'un modèle spécifique"
    )
    
    parser.add_argument(
        "--list-models", 
        action="store_true",
        help="Affiche les modèles disponibles et quitte"
    )
    
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Active le mode verbeux"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Taille des lots pour le traitement (surcharge la config)"
    )
    
    return parser


def display_available_models():
    """Affiche les modèles disponibles."""
    print("\n🤖 MODÈLES DISPONIBLES:")
    print("=" * 50)
    
    models = get_available_models()
    
    for model_id, info in models.items():
        status_icon = "✅" if info["enabled"] and info["available"] else "❌"
        primary_icon = "🎯" if info["is_primary"] else "  "
        
        print(f"{status_icon} {primary_icon} {info['name']}")
        print(f"      ID: {info['model_id']}")
        print(f"      Activé: {'Oui' if info['enabled'] else 'Non'}")
        print(f"      Disponible: {'Oui' if info['available'] else 'Non'}")
        print()


def display_result(result: dict):
    """Affiche le résultat d'analyse de manière formatée."""
    print("\n" + "="*80)
    print("🧠 RÉSULTAT DE L'ANALYSE DE SURVEILLANCE")
    print("="*80)
    
    # Métadonnées
    print(f"📍 Section: {result['section']}")
    print(f"🕐 Timestamp: {result['timestamp']}")
    print(f"⚡ Durée: {result['metadata']['analysis_duration']:.2f}s")
    print(f"🤖 Modèle utilisé: {result['metadata']['model_used']}")
    
    # Analyse VLM
    if result.get('thinking'):
        print(f"\n🧠 RÉFLEXION DU VLM:")
        print("-" * 40)
        print(result['thinking'])
    
    print(f"\n📊 DESCRIPTION DE LA SCÈNE:")
    print("-" * 40)
    print(result['summary'])
    
    # Décision LLM
    decision = result['decision']
    print(f"\n⚖️  DÉCISION DE L'ORCHESTRATEUR:")
    print("-" * 40)
    print(f"Niveau de suspicion: {decision['suspicion_level'].upper()}")
    print(f"Type d'alerte: {decision['alert_type']}")
    print(f"Action recommandée: {decision['action']}")
    print(f"Confiance: {decision['confidence']:.2f}")
    
    print(f"\n💭 RAISONNEMENT:")
    print(decision['reasoning'])
    
    if decision.get('recommended_tools'):
        print(f"\n🛠️  OUTILS RECOMMANDÉS:")
        for tool in decision['recommended_tools']:
            print(f"  • {tool}")
    
    # Indicateur de risque
    risk_level = decision['suspicion_level']
    if risk_level == "high":
        print(f"\n🚨 ATTENTION: NIVEAU DE RISQUE ÉLEVÉ!")
    elif risk_level == "medium":
        print(f"\n⚠️  Surveillance renforcée recommandée")
    else:
        print(f"\n✅ Situation normale")
    
    print("="*80)


def main():
    """Fonction principale."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Configuration du logging
    if args.verbose:
        settings.config.log_level = "DEBUG"
    
    logger = get_surveillance_logger()
    
    # Affichage des modèles et sortie
    if args.list_models:
        display_available_models()
        return 0
    
    # Validation du chemin vidéo
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"❌ Erreur: La vidéo '{video_path}' n'existe pas.")
        return 1
    
    try:
        logger.info("🎬 Démarrage du Surveillance Orchestrator")
        logger.info(f"Version Python: {sys.version}")
        
        # Configuration optionnelle
        if args.batch_size:
            settings.config.batch_size = args.batch_size
            logger.info(f"Taille de lot configurée: {args.batch_size}")
        
        # Sélection du modèle si spécifié
        if args.model:
            from src.config import ModelType
            if args.model == "smolvlm":
                settings.set_primary_vlm(ModelType.SMOLVLM)
                logger.info("Modèle forcé: SmolVLM")
            elif args.model == "kim":
                if settings.get_model_config(ModelType.KIM).enabled:
                    settings.set_primary_vlm(ModelType.KIM)
                    logger.info("Modèle forcé: KIM")
                else:
                    logger.warning("KIM n'est pas activé, utilisation de SmolVLM")
        
        # Affichage de la configuration
        print(f"\n🎯 ANALYSE EN COURS...")
        print(f"📹 Vidéo: {video_path.name}")
        print(f"📍 Section: {args.section}")
        print(f"🕐 Moment: {args.time_of_day}")
        print(f"👥 Affluence: {args.crowd_density}")
        print(f"🎞️  Mode: {'Keyframes' if args.keyframes else 'Frames régulières'}")
        print(f"🤖 VLM principal: {settings.config.primary_vlm.value}")
        
        # Initialisation de l'orchestrateur
        orchestrator = SurveillanceOrchestrator()
        
        # Analyse de la vidéo
        result = orchestrator.analyze(
            video_path=video_path,
            section=args.section,
            time_of_day=args.time_of_day,
            crowd_density=args.crowd_density,
            use_keyframes=args.keyframes
        )
        
        if result:
            display_result(result)
            
            # Affichage des statistiques de session
            stats = orchestrator.get_session_stats()
            print(f"\n📊 STATISTIQUES DE SESSION:")
            print(f"Analyses: {stats['analyses_count']}")
            print(f"Alertes: {stats['alerts_count']}")
            print(f"Taux d'alerte: {stats['alert_rate']}%")
            
            return 0
        else:
            print("❌ L'analyse a échoué.")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Interruption par l'utilisateur")
        return 130
    except Exception as e:
        logger.error(f"Erreur inattendue: {e}")
        return 1
    finally:
        # Nettoyage
        try:
            if 'orchestrator' in locals():
                orchestrator.cleanup()
        except:
            pass


if __name__ == "__main__":
    exit(main())