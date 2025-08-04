from src.models.phi3_wrapper import Phi3Wrapper
import json

def test_phi3():
    context = {
        "section": "Rayon électronique",
        "time": "Soirée",
        "density": "faible",
        "vlm_analysis": "Une personne semble cacher un produit dans sa veste.",
        "last_alerts": ["Alerte dissimulation il y a 10 min."]
    }

    phi = Phi3Wrapper()
    decision = phi.analyze(context)

    print("\n🧠 Décision de Phi-3 :")
    print(json.dumps(decision, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test_phi3()
