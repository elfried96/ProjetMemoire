# src/orchestrator/controller.py

import json
from pathlib import Path
from datetime import datetime
from src.utils.preprocessing import extract_frames
from src.models.smolvlm_wrapper import SmolVLMWrapper
from src.models.phi3_wrapper import Phi3Wrapper


class SurveillanceOrchestrator:
    def __init__(self):
        self.vlm = SmolVLMWrapper()
        self.llm = Phi3Wrapper()
        self.memory_path = Path("memory/orchestration.json")
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.memory_path.exists():
            self.memory_path.write_text("[]", encoding="utf-8")

    def get_recent_alerts(self, section: str, limit: int = 3):
        try:
            history = json.loads(self.memory_path.read_text(encoding="utf-8"))
            section_history = [h for h in history if h.get("section") == section]
            return [f"{h['llm_decision']['alert_type']} - {h['llm_decision']['suspicion_level']} ({h['timestamp']})"
                    for h in section_history[-limit:]]
        except Exception as e:
            print("⚠️ Erreur lecture mémoire :", e)
            return []

    def analyze(self, video_path: str, section: str, time_of_day: str, crowd_density: str):
        # 1. Extraire les frames de la vidéo
        frames = extract_frames(video_path)
        if not frames:
            print("❌ Aucune frame extraite.")
            return None

        # 2. Appel VLM
        vlm_result = self.vlm.analyze_frames(frames, section, time_of_day, crowd_density)
        vlm_description = vlm_result.get("summary", "")

        # 3. Construction contexte pour le LLM (Phi3)
        alerts = self.get_recent_alerts(section)
        context = {
            "section": section,
            "time": time_of_day,
            "density": crowd_density,
            "vlm_analysis": vlm_description,
            "last_alerts": alerts
        }

        # 4. Appel Phi3 (raisonnement)
        decision = self.llm.analyze(context)

        # 5. Construction d’un événement
        event = {
            "timestamp": datetime.now().isoformat(),
            "section": section,
            "vlm_summary": vlm_description,
            "llm_decision": decision
        }

        # 6. Mise à jour de la mémoire globale
        try:
            history = json.loads(self.memory_path.read_text(encoding="utf-8"))
            history.append(event)
            self.memory_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            print("⚠️ Erreur écriture mémoire :", e)

        # 7. Sauvegarde séparée dans outputs/log_decisions.json
        log_dir = Path("outputs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "log_decisions.json"
        if not log_file.exists():
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump([], f)

        with open(log_file, "r+", encoding="utf-8") as f:
            existing = json.load(f)
            existing.append(event)
            f.seek(0)
            json.dump(existing, f, indent=2, ensure_ascii=False)

        # 8. Retour du résultat structuré
        return {
            "thinking": vlm_result.get("thinking", ""),
            "summary": vlm_description,
            "decision": decision
        }
