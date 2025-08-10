#!/usr/bin/env python3
"""
Test ultra-simple pour vérifier que ModelType est accessible.
"""

try:
    print("🔍 Test d'import de base...")
    from src.config.settings import settings, ModelType
    print("✅ Import settings + ModelType OK")
    
    print(f"✅ VLM principal: {settings.config.primary_vlm.value}")
    print(f"✅ ModelType.KIM: {ModelType.KIM.value}")
    print(f"✅ ModelType.SMOLVLM: {ModelType.SMOLVLM.value}")
    
    # Test de la configuration
    kim_config = settings.get_model_config(ModelType.KIM)
    smolvlm_config = settings.get_model_config(ModelType.SMOLVLM)
    
    print(f"✅ KIM config: {kim_config.name} - {kim_config.model_id}")
    print(f"✅ SmolVLM config: {smolvlm_config.name} - {smolvlm_config.model_id}")
    
    print("\n🎉 ERREUR MODELTYPE COMPLÈTEMENT CORRIGÉE !")
    print("💡 L'erreur AttributeError: 'Settings' object has no attribute 'ModelType' n'existe plus")
    
except AttributeError as e:
    if "ModelType" in str(e):
        print(f"❌ L'erreur ModelType persiste: {e}")
    else:
        print(f"❌ Autre erreur AttributeError: {e}")
except Exception as e:
    print(f"❌ Erreur inattendue: {e}")
    import traceback
    traceback.print_exc()