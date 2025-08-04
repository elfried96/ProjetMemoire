# src/models/phi3_wrapper.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

class Phi3Wrapper:
    def __init__(self, model_name="microsoft/phi-2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

    def build_prompt(self, context: dict) -> str:
        """
        Construit un prompt textuel à partir du contexte d'analyse.
        """
        return f"""
Tu es une intelligence stratégique de surveillance.
Voici le résumé de la scène :
- Section : {context['section']}
- Heure : {context['time']}
- Affluence : {context['density']}
- Analyse visuelle : {context['vlm_analysis']}
- Alertes précédentes : {', '.join(context.get('last_alerts', []))}

Tâche :
Analyse la scène et fournis une réponse JSON structurée comme suit :
{{
  "suspicion_level": "low/medium/high",
  "alert_type": "rien / dissimulation / repérage / tentative vol",
  "reasoning": "raisonnement étape par étape",
  "action": "rien / alerter / surveiller discretement / intervenir"
}}
        """.strip()

    def analyze(self, context: dict) -> dict:
        """
        Effectue une génération basée sur le prompt et retourne un JSON structuré.
        """
        prompt = self.build_prompt(context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # 🧠 Extraction du JSON depuis la réponse
        try:
            json_start = decoded.find("{")
            json_str = decoded[json_start:]
            return json.loads(json_str)
        except Exception as e:
            return {
                "suspicion_level": "unknown",
                "alert_type": "inconnu",
                "reasoning": "❌ Erreur extraction JSON : " + str(e),
                "action": "manuelle"
            }
