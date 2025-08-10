#!/usr/bin/env python3
"""
Test rapide pour vérifier que l'erreur ModelType est corrigée.
"""

try:
    print("🔍 Test d'import des modules...")
    from src.config.settings import settings, ModelType
    print("✅ Import settings + ModelType OK")
    
    from src.models.kim_wrapper import KIMWrapper
    print("✅ Import KIMWrapper OK")
    
    from src.models import get_available_models
    print("✅ Import get_available_models OK")
    
    print("\n🧪 Test des fonctions...")
    
    # Test 1: Configuration ModelType
    print(f"VLM principal: {settings.config.primary_vlm.value}")
    print(f"ModelType.KIM: {ModelType.KIM.value}")
    
    # Test 2: Configuration KIM
    kim_config = settings.get_model_config(ModelType.KIM)
    print(f"KIM activé: {kim_config.enabled}")
    print(f"KIM model_id: {kim_config.model_id}")
    
    # Test 3: KIMWrapper.is_available() - La fonction qui causait l'erreur
    print(f"KIM disponible: {KIMWrapper.is_available()}")
    
    # Test 4: get_available_models() - La fonction qui appelait is_available()
    models = get_available_models()
    print(f"Modèles trouvés: {list(models.keys())}")
    
    print("\n🎉 TOUS LES TESTS RÉUSSIS !")
    print("✅ L'erreur ModelType est corrigée")
    
except Exception as e:
    print(f"❌ Erreur détectée: {e}")
    import traceback
    traceback.print_exc()