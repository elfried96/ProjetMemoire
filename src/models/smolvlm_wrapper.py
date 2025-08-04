import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import gc

class SmolVLMWrapper:
    def __init__(self, model_name="nv-tlabs/smol-vlm", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Chargement du processor et du modèle
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        print(f"✅ Modèle SmolVLM chargé sur {self.device}.")

        # Gabarit de prompt
        self.prompt_template = """
Vous êtes un système de surveillance intelligente. Analysez ces images issues de caméras de sécurité.
CONTEXTE :
- Section : {section}
- Heure : {time}
- Affluence : {density}

TÂCHES :
1. Décrivez les actions visibles
2. Soulignez tout comportement suspect
3. Produisez une synthèse utile pour un agent de sécurité

◁think▷
        """.strip()

    def generate_prompt(self, section, time_of_day, crowd_density):
        return self.prompt_template.format(
            section=section,
            time=time_of_day,
            density=crowd_density
        )

    def analyze(self, images, prompt):
        """
        Analyse une liste d'images avec un prompt texte.
        Args:
            images (List[PIL.Image]): images à analyser
            prompt (str): description de la tâche ou du contexte
        Returns:
            dict: pensée, résumé, texte brut
        """
        inputs = self.processor(
            images=images,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )

        decoded = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        thinking, summary = self.extract_thinking_and_summary(decoded)
        return {
            "thinking": thinking,
            "summary": summary,
            "raw": decoded
        }

    def extract_thinking_and_summary(self, text, bot="◁think▷", eot="◁/think▷"):
        """
        Extrait les sections ◁think▷ ... ◁/think▷ du texte généré.
        """
        if eot in text:
            thinking = text[text.find(bot) + len(bot):text.find(eot)].strip()
            summary = text[text.find(eot) + len(eot):].strip()
            return thinking, summary
        return "", text

    def analyze_frames(self, frames, section, time_of_day, crowd_density, batch_size=4):
        """
        Analyse des frames par lot pour éviter l'overflow mémoire.

        Args:
            frames: liste d'images PIL
            section: zone du magasin
            time_of_day: moment de la journée
            crowd_density: affluence
            batch_size: nombre d’images par lot
        Returns:
            dict: pensée globale, résumé global, texte brut
        """
        prompt = self.generate_prompt(section, time_of_day, crowd_density)

        all_thoughts = []
        all_summaries = []
        all_raw = []

        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            result = self.analyze(batch, prompt)

            if result["thinking"]:
                all_thoughts.append(result["thinking"])
            if result["summary"]:
                all_summaries.append(result["summary"])
            all_raw.append(result["raw"])

        combined_thinking = "\n\n".join(all_thoughts)
        combined_summary = "\n".join(all_summaries)

        return {
            "thinking": combined_thinking,
            "summary": combined_summary,
            "raw": "\n---\n".join(all_raw)
        }

    def cleanup(self):
        """
        Libération de la mémoire GPU
        """
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
        print("🧹 Mémoire libérée.")
