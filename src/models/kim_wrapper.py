import torch
import gc
import cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from utils.preprocessing import prepr
class KIMWrapper:
    def __init__(self, model_name="moonshotai/Kimi-VL-A3B-Thinking-2506"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.prompt_template = """
Vous êtes un système de surveillance intelligente. Analysez cette image de magasin.
CONTEXTE: {store_section}, {time_of_day}, {crowd_density}
TÂCHES:
1. Identifiez toutes les personnes et leurs actions
2. Détectez tout comportement inhabituel ou suspect
3. Déterminez quels outils utiliser pour validation:
- object_detection: pour identifier les objets manipulés
- movement_analysis: pour analyser les déplacements
- context_validation: pour valider dans le contexte du magasin
CRITÈRES SUSPECTS:
- Dissimulation d'objets
- Regards furtifs répétés
- Mouvements vers les sorties sans passer en caisse
- Manipulation d'étiquettes/emballages
RÉPONSE ATTENDUE:
{{
"analysis": "description détaillée",
"suspicion_level": "low/medium/high",
"recommended_tools": ["tool1", "tool2"],
"reasoning": "justification"
}}
        """.strip()

    def extract_frames(self, video_path, seconds_per_frame=2, max_frames=5):
        vidcap = cv2.VideoCapture(video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * seconds_per_frame)

        frames = []
        success, image = vidcap.read()
        count = 0
        while success and len(frames) < max_frames:
            if count % frame_interval == 0:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                preprocessed_image = preprocess_frame(pil_image)
                frames.append(preprocessed_image)
            success, image = vidcap.read()
            count += 1

        vidcap.release()
        print(f"✅ {len(frames)} images extraites de la vidéo.")
        return frames

    def generate_prompt(self, section, time_of_day, crowd_density):
        return self.prompt_template.format(
            store_section=section,
            time_of_day=time_of_day,
            crowd_density=crowd_density
        )

    def extract_thinking_and_summary(self, text, bot="◁think▷", eot="◁/think▷"):
        if eot in text:
            thinking = text[text.find(bot) + len(bot):text.find(eot)].strip()
            summary = text[text.find(eot) + len(eot):].strip()
            return thinking, summary
        return "", text

    def analyze_video(self, video_path, section, time_of_day, crowd_density):
        try:
            frames = self.extract_frames(video_path)
            if not frames:
                print("❌ Aucune frame extraite.")
                return None

            prompt = self.generate_prompt(section, time_of_day, crowd_density)

            inputs = self.processor(
                images=frames,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=1024, temperature=0.7)

            trimmed_ids = generated_ids[:, inputs.input_ids.shape[1]:]
            decoded = self.processor.batch_decode(
                trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            thinking, summary = self.extract_thinking_and_summary(decoded)
            return {
                "thinking": thinking,
                "summary": summary,
                "raw": decoded
            }

        finally:
            self.cleanup()

    def cleanup(self):
        print("🧹 Libération des ressources...")
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("✅ Mémoire GPU/CPU nettoyée.")
