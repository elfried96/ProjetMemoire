#!/usr/bin/env python3
"""Script de test pour vérifier les corrections appliquées."""

import os
import sys
import torch
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_cuda_setup():
    """Test la configuration CUDA."""
    print("🔍 Test Configuration CUDA:")
    print(f"   CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   Devices CUDA: {torch.cuda.device_count()}")
        print(f"   Device actuel: {torch.cuda.current_device()}")
        print(f"   Nom GPU: {torch.cuda.get_device_name()}")
        print(f"   Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory // 1e9:.1f}GB")
    print()

def test_smolvlm_device():
    """Test du chargement SmolVLM sans erreur de device."""
    try:
        print("🔍 Test SmolVLM Device Loading:")
        from models.smolvlm_wrapper import SmolVLMWrapper
        
        # Test avec un modèle léger pour vérifier la logique de device
        wrapper = SmolVLMWrapper("microsoft/DialoGPT-small", "cuda")
        print("   ✅ SmolVLM wrapper initialisé sans erreur")
        wrapper.cleanup()
        print("   ✅ Cleanup réussi")
        
    except Exception as e:
        print(f"   ❌ Erreur SmolVLM: {e}")
    print()

def test_phi3_device():
    """Test du chargement Phi3 sans erreur de device."""
    try:
        print("🔍 Test Phi3 Device Loading:")
        from models.phi3_wrapper import Phi3Wrapper
        
        # Test avec un modèle léger
        wrapper = Phi3Wrapper("microsoft/DialoGPT-small", "cuda") 
        print("   ✅ Phi3 wrapper initialisé sans erreur")
        wrapper.cleanup()
        print("   ✅ Cleanup réussi")
        
    except Exception as e:
        print(f"   ❌ Erreur Phi3: {e}")
    print()

def test_json_parsing():
    """Test l'amélioration du parsing JSON."""
    try:
        print("🔍 Test JSON Parsing:")
        from models.phi3_wrapper import Phi3Wrapper
        
        wrapper = Phi3Wrapper()
        
        # Test avec différents formats JSON problématiques
        test_cases = [
            '{"suspicion_level": "low", "alert_type": "observation"}',
            'Voici ma réponse: {"suspicion_level": "medium"} avec du texte après',
            '{\n  "suspicion_level": "high",\n  "alert_type": "vol"\n}',
        ]
        
        for i, test_json in enumerate(test_cases, 1):
            try:
                result = wrapper._extract_and_validate_json(test_json)
                print(f"   ✅ Test {i}: JSON parsé → {result.suspicion_level}")
            except Exception as e:
                print(f"   ❌ Test {i}: {e}")
                
    except Exception as e:
        print(f"   ❌ Erreur test JSON: {e}")
    print()

if __name__ == "__main__":
    print("🧪 TEST DES CORRECTIONS APPLIQUÉES")
    print("=" * 50)
    
    # Charger les variables d'environnement
    from dotenv import load_dotenv
    load_dotenv()
    
    test_cuda_setup()
    test_smolvlm_device() 
    test_phi3_device()
    test_json_parsing()
    
    print("✅ Tests terminés!")