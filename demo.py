#!/usr/bin/env python3
"""
Script de démonstration du Surveillance Orchestrator.
Montre les capacités du système avec une interface interactive.
"""

import sys
from pathlib import Path
from typing import List

from src.orchestrator.controller import SurveillanceOrchestrator
from src.config import settings, ModelType
from src.models import get_available_models
from src.utils.logging import get_surveillance_logger

logger = get_surveillance_logger()


class SurveillanceDemo:
    """Classe de démonstration interactive."""
    
    def __init__(self):
        self.orchestrator = None
        self.video_files = self._find_demo_videos()
    
    def _find_demo_videos(self) -> List[Path]:
        """Trouve les vidéos de démonstration disponibles."""
        video_dir = Path("videos")
        if not video_dir.exists():
            return []
        
        extensions = {'.mp4', '.avi', '.mov', '.mkv'}
        videos = []
        
        for ext in extensions:
            videos.extend(video_dir.glob(f"*{ext}"))
        
        return sorted(videos)[:5]  # Max 5 vidéos pour la demo
    
    def display_welcome(self):
        """Affiche le message de bienvenue."""
        print("="*80)
        print("SURVEILLANCE ORCHESTRATOR - DÉMONSTRATION")
        print("="*80)
        print()
        print("Système de surveillance intelligente utilisant l'IA")
        print("Analyse automatique de vidéos de surveillance")
        print("Détection de comportements suspects avec VLM + LLM")
        print()
        
        # Affichage de la configuration actuelle
        models = get_available_models()
        primary_model = None
        
        for model_id, info in models.items():
            if info["is_primary"]:
                primary_model = info["name"]
                break
        
        print(f"Modèle principal: {primary_model}")
        print(f"Mémoire nettoyée après analyse: {'Oui' if settings.config.cleanup_after_analysis else 'Non'}")
        print()
    
    def display_menu(self):
        """Affiche le menu principal."""
        print("MENU PRINCIPAL")
        print("-" * 30)
        print("1. Analyser une vidéo")
        print("2. Analyser toutes les vidéos de démo")
        print("3. Changer de modèle VLM")
        print("4. Afficher les statistiques")
        print("5. Test des capacités système")
        print("6. Quitter")
        print()
    
    def display_videos(self) -> bool:
        """Affiche les vidéos disponibles."""
        if not self.video_files:
            print("Aucune vidéo trouvée dans le dossier 'videos/'")
            print("Ajoutez des fichiers vidéo (.mp4, .avi, .mov, .mkv) dans ce dossier")
            return False
        
        print("VIDÉOS DISPONIBLES:")
        print("-" * 30)
        for i, video in enumerate(self.video_files, 1):
            size_mb = video.stat().st_size / (1024 * 1024)
            print(f"{i}. {video.name} ({size_mb:.1f} MB)")
        print()
        return True
    
    def choose_video(self) -> Path:
        """Permet à l'utilisateur de choisir une vidéo."""
        if not self.display_videos():
            return None
        
        while True:
            try:
                choice = input("Choisissez une vidéo (numéro) ou 'q' pour retour: ").strip()
                
                if choice.lower() == 'q':
                    return None
                
                idx = int(choice) - 1
                if 0 <= idx < len(self.video_files):
                    return self.video_files[idx]
                else:
                    print("Choix invalide. Essayez encore.")
                    
            except ValueError:
                print("❌ Veuillez entrer un numéro valide.")
    
    def get_analysis_params(self):
        """Récupère les paramètres d'analyse de l'utilisateur."""
        print("⚙️  PARAMÈTRES D'ANALYSE")
        print("-" * 30)
        
        sections = ["Rayon cosmétique", "Entrée principale", "Caisse", "Rayon électronique", "Sortie"]
        times = ["Matin", "Après-midi", "Fin d'après-midi", "Soirée"]
        densities = ["faible", "modérée", "dense"]
        
        print("Sections disponibles:", ", ".join(sections))
        section = input(f"Section (défaut: {sections[0]}): ").strip() or sections[0]
        
        print("Moments disponibles:", ", ".join(times))
        time_of_day = input(f"Moment (défaut: {times[2]}): ").strip() or times[2]
        
        print("Affluence disponible:", ", ".join(densities))
        crowd_density = input(f"Affluence (défaut: {densities[2]}): ").strip() or densities[2]
        
        keyframes = input("Utiliser l'extraction intelligente de keyframes? (y/N): ").strip().lower() == 'y'
        
        return section, time_of_day, crowd_density, keyframes
    
    def analyze_single_video(self):
        """Analyse une seule vidéo choisie par l'utilisateur."""
        video = self.choose_video()
        if not video:
            return
        
        section, time_of_day, crowd_density, keyframes = self.get_analysis_params()
        
        print(f"\n🎬 ANALYSE DE: {video.name}")
        print("=" * 50)
        
        if not self.orchestrator:
            self.orchestrator = SurveillanceOrchestrator()
        
        result = self.orchestrator.analyze(
            video_path=video,
            section=section,
            time_of_day=time_of_day,
            crowd_density=crowd_density,
            use_keyframes=keyframes
        )
        
        if result:
            self._display_result_summary(result)
        else:
            print("❌ Échec de l'analyse")
    
    def analyze_all_videos(self):
        """Analyse toutes les vidéos disponibles avec des paramètres aléatoires."""
        if not self.video_files:
            print("❌ Aucune vidéo disponible")
            return
        
        import random
        
        sections = ["Rayon cosmétique", "Entrée principale", "Caisse", "Rayon électronique"]
        times = ["Matin", "Après-midi", "Soirée"]
        densities = ["faible", "modérée", "dense"]
        
        print(f"🎬 ANALYSE DE TOUTES LES VIDÉOS ({len(self.video_files)})")
        print("=" * 60)
        
        if not self.orchestrator:
            self.orchestrator = SurveillanceOrchestrator()
        
        results = []
        
        for i, video in enumerate(self.video_files, 1):
            print(f"\n📹 Analyse {i}/{len(self.video_files)}: {video.name}")
            
            # Paramètres aléatoires pour la diversité
            section = random.choice(sections)
            time_of_day = random.choice(times)
            crowd_density = random.choice(densities)
            keyframes = random.choice([True, False])
            
            result = self.orchestrator.analyze(
                video_path=video,
                section=section,
                time_of_day=time_of_day,
                crowd_density=crowd_density,
                use_keyframes=keyframes
            )
            
            if result:
                results.append(result)
                decision = result['decision']
                print(f"   Résultat: {decision['suspicion_level']} - {decision['alert_type']}")
            else:
                print("   ❌ Échec")
        
        # Résumé final
        print(f"\n📊 RÉSUMÉ GLOBAL")
        print("-" * 30)
        print(f"Vidéos analysées: {len(results)}/{len(self.video_files)}")
        
        if results:
            suspicion_counts = {}
            for result in results:
                level = result['decision']['suspicion_level']
                suspicion_counts[level] = suspicion_counts.get(level, 0) + 1
            
            for level, count in suspicion_counts.items():
                print(f"{level.capitalize()}: {count}")
    
    def switch_model(self):
        """Permet de changer de modèle VLM."""
        print("🔄 CHANGEMENT DE MODÈLE")
        print("-" * 30)
        
        models = get_available_models()
        available_models = []
        
        print("Modèles disponibles:")
        for i, (model_id, info) in enumerate(models.items(), 1):
            status = "✅" if info["enabled"] and info["available"] else "❌"
            primary = " (ACTUEL)" if info["is_primary"] else ""
            print(f"{i}. {status} {info['name']}{primary}")
            if info["enabled"] and info["available"]:
                available_models.append(model_id)
        
        if len(available_models) <= 1:
            print("⚠️  Un seul modèle disponible, aucun changement possible.")
            return
        
        try:
            choice = int(input("\nChoisissez un modèle (numéro): "))
            model_ids = list(models.keys())
            
            if 1 <= choice <= len(model_ids):
                selected_model = model_ids[choice - 1]
                
                if selected_model == "kim":
                    if self.orchestrator:
                        success = self.orchestrator.switch_to_kim()
                        if success:
                            print("✅ Basculement vers KIM réussi")
                        else:
                            print("❌ Impossible de basculer vers KIM")
                    else:
                        print("⚠️  Orchestrateur non initialisé")
                
                elif selected_model == "smolvlm":
                    if self.orchestrator:
                        self.orchestrator.switch_to_smolvlm()
                        print("✅ Basculement vers SmolVLM")
                    else:
                        settings.set_primary_vlm(ModelType.SMOLVLM)
                        print("✅ SmolVLM défini comme principal")
            else:
                print("❌ Choix invalide")
                
        except ValueError:
            print("❌ Veuillez entrer un numéro valide")
    
    def show_statistics(self):
        """Affiche les statistiques du système."""
        print("📊 STATISTIQUES SYSTÈME")
        print("-" * 30)
        
        if self.orchestrator:
            stats = self.orchestrator.get_session_stats()
            memory_stats = self.orchestrator.memory.get_statistics()
            
            print("Session actuelle:")
            print(f"  Analyses: {stats['analyses_count']}")
            print(f"  Alertes: {stats['alerts_count']}")
            print(f"  Taux d'alerte: {stats['alert_rate']}%")
            print(f"  Durée session: {stats['session_duration']}")
            print(f"  Changements de modèle: {stats['model_switches']}")
            
            print("\nMémoire globale:")
            print(f"  Total analyses: {memory_stats.get('total_analyses', 0)}")
            print(f"  Total alertes: {memory_stats.get('total_alerts', 0)}")
        else:
            print("⚠️  Aucune session active")
    
    def test_capabilities(self):
        """Test des capacités système."""
        print("🧪 TEST DES CAPACITÉS")
        print("-" * 30)
        
        # Test GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                print("❌ Pas de GPU CUDA disponible")
        except ImportError:
            print("❌ PyTorch non installé")
        
        # Test des modèles
        print("\nTest des modèles:")
        try:
            from src.models import create_vlm_model, create_llm_model
            
            vlm = create_vlm_model()
            print(f"✅ VLM: {type(vlm).__name__}")
            
            llm = create_llm_model()
            print(f"✅ LLM: {type(llm).__name__}")
            
        except Exception as e:
            print(f"❌ Erreur modèles: {e}")
        
        # Test preprocessing
        print(f"\nVidéos de test: {len(self.video_files)}")
    
    def _display_result_summary(self, result: dict):
        """Affiche un résumé du résultat d'analyse."""
        decision = result['decision']
        metadata = result['metadata']
        
        print(f"\n✅ ANALYSE TERMINÉE")
        print(f"Durée: {metadata['analysis_duration']:.2f}s")
        print(f"Modèle: {metadata['model_used']}")
        print()
        print(f"🎯 DÉCISION: {decision['suspicion_level'].upper()}")
        print(f"Type: {decision['alert_type']}")
        print(f"Action: {decision['action']}")
        print(f"Confiance: {decision['confidence']:.2f}")
        print()
        print(f"💭 Raisonnement:")
        print(f"   {decision['reasoning']}")
        
        # Affichage des détails d'analyse si disponibles
        if 'vlm_result' in result and result['vlm_result']:
            vlm_result = result['vlm_result']
            print(f"\n🔍 ANALYSE VLM DÉTAILLÉE:")
            print("=" * 40)
            if hasattr(vlm_result, 'thinking') and vlm_result.thinking:
                print(f"💭 Réflexion du modèle:")
                print(f"   {vlm_result.thinking}")
                print()
            if hasattr(vlm_result, 'summary') and vlm_result.summary:
                print(f"📋 Résumé VLM:")
                print(f"   {vlm_result.summary}")
                print()
            if hasattr(vlm_result, 'raw_output') and vlm_result.raw_output and vlm_result.raw_output != vlm_result.summary:
                print(f"🤖 Sortie brute du modèle:")
                print(f"   {vlm_result.raw_output[:500]}{'...' if len(vlm_result.raw_output) > 500 else ''}")
                print()
        
        if 'llm_result' in result and result['llm_result']:
            llm_result = result['llm_result']
            print(f"\n🧠 ANALYSE LLM DÉTAILLÉE:")
            print("=" * 40)
            if hasattr(llm_result, 'thinking') and llm_result.thinking:
                print(f"💭 Réflexion du modèle:")
                print(f"   {llm_result.thinking}")
                print()
            if hasattr(llm_result, 'raw_output') and llm_result.raw_output:
                print(f"🤖 Sortie brute du modèle:")
                print(f"   {llm_result.raw_output[:500]}{'...' if len(llm_result.raw_output) > 500 else ''}")
                print()
    
    def run(self):
        """Lance la démonstration interactive."""
        self.display_welcome()
        
        while True:
            try:
                self.display_menu()
                choice = input("Votre choix (1-6): ").strip()
                
                if choice == "1":
                    self.analyze_single_video()
                elif choice == "2":
                    self.analyze_all_videos()
                elif choice == "3":
                    self.switch_model()
                elif choice == "4":
                    self.show_statistics()
                elif choice == "5":
                    self.test_capabilities()
                elif choice == "6":
                    break
                else:
                    print("❌ Choix invalide. Essayez encore.")
                
                input("\n▶️  Appuyez sur Entrée pour continuer...")
                print("\n" + "="*80)
                
            except KeyboardInterrupt:
                print("\n\n👋 Au revoir!")
                break
            except Exception as e:
                logger.error(f"Erreur dans la démo: {e}")
                print(f"❌ Erreur: {e}")
        
        # Nettoyage
        if self.orchestrator:
            self.orchestrator.cleanup()


def main():
    """Fonction principale."""
    demo = SurveillanceDemo()
    demo.run()


if __name__ == "__main__":
    main()