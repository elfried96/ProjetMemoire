"""
Utilitaires de préprocessing pour la surveillance intelligente.
Extraction et amélioration d'images à partir de vidéos.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union
from PIL import Image, ImageEnhance, ImageFilter
import logging

from ..config import settings
from .logging import get_surveillance_logger

logger = get_surveillance_logger()


class VideoProcessor:
    """Processeur vidéo avec gestion d'erreurs robuste."""
    
    def __init__(self):
        self.supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    
    def is_video_supported(self, video_path: Union[str, Path]) -> bool:
        """
        Vérifie si le format vidéo est supporté.
        
        Args:
            video_path: Chemin vers la vidéo
            
        Returns:
            True si le format est supporté
        """
        path = Path(video_path)
        return path.suffix.lower() in self.supported_formats
    
    def get_video_info(self, video_path: Union[str, Path]) -> dict:
        """
        Récupère les informations d'une vidéo.
        
        Args:
            video_path: Chemin vers la vidéo
            
        Returns:
            Dictionnaire avec les infos de la vidéo
        """
        path = Path(video_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Vidéo non trouvée: {path}")
        
        if not self.is_video_supported(path):
            raise ValueError(f"Format non supporté: {path.suffix}")
        
        try:
            cap = cv2.VideoCapture(str(path))
            
            if not cap.isOpened():
                raise ValueError(f"Impossible d'ouvrir la vidéo: {path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            info = {
                "path": str(path),
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
                "resolution": (width, height),
                "size_mb": path.stat().st_size / (1024 * 1024)
            }
            
            logger.info(f"Vidéo analysée: {duration:.1f}s, {fps:.1f}fps, {width}x{height}")
            return info
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse vidéo: {e}")
            raise
    
    def extract_frames(
        self,
        video_path: Union[str, Path],
        seconds_per_frame: Optional[float] = None,
        max_frames: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Extrait des frames d'une vidéo avec gestion d'erreurs robuste.
        
        Args:
            video_path: Chemin vers la vidéo
            seconds_per_frame: Intervalle entre frames (utilise config si None)
            max_frames: Nombre max de frames (utilise config si None)
            
        Returns:
            Liste d'images PIL prétraitées
        """
        if seconds_per_frame is None:
            seconds_per_frame = settings.config.processing.seconds_per_frame
        
        if max_frames is None:
            max_frames = settings.config.processing.max_frames
        
        path = Path(video_path)
        video_info = self.get_video_info(path)
        
        logger.info(
            f"Extraction frames: {seconds_per_frame}s/frame, max {max_frames} frames"
        )
        
        frames = []
        cap = None
        
        try:
            cap = cv2.VideoCapture(str(path))
            fps = video_info["fps"]
            frame_interval = max(1, int(fps * seconds_per_frame))
            
            success, image = cap.read()
            frame_count = 0
            extracted_count = 0
            
            while success and extracted_count < max_frames:
                if frame_count % frame_interval == 0:
                    # Conversion BGR -> RGB
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_image)
                    
                    # Préprocessing de l'image
                    processed_image = self.preprocess_frame(pil_image)
                    frames.append(processed_image)
                    extracted_count += 1
                    
                    if extracted_count % 10 == 0:
                        logger.debug(f"Frames extraites: {extracted_count}/{max_frames}")
                
                success, image = cap.read()
                frame_count += 1
            
            logger.info(f"✅ {len(frames)} frames extraites et prétraitées")
            return frames
            
        except Exception as e:
            logger.error(f"❌ Erreur extraction frames: {e}")
            raise
        finally:
            if cap:
                cap.release()
    
    def preprocess_frame(
        self,
        image: Image.Image,
        target_size: Optional[Tuple[int, int]] = None,
        contrast: Optional[float] = None,
        brightness: Optional[float] = None,
        denoise: Optional[bool] = None
    ) -> Image.Image:
        """
        Préprocesse une image pour améliorer la perception VLM.
        
        Args:
            image: Image PIL à traiter
            target_size: Taille cible (utilise config si None)
            contrast: Facteur de contraste (utilise config si None)
            brightness: Facteur de luminosité (utilise config si None)
            denoise: Activer réduction bruit (utilise config si None)
            
        Returns:
            Image PIL prétraitée
        """
        config = settings.config.processing
        
        if target_size is None:
            target_size = config.target_size
        if contrast is None:
            contrast = config.contrast
        if brightness is None:
            brightness = config.brightness
        if denoise is None:
            denoise = config.denoise
        
        try:
            # Redimensionnement intelligent (garde le ratio)
            if target_size != image.size:
                image = self._smart_resize(image, target_size)
            
            # Amélioration du contraste
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(contrast)
            
            # Amélioration de la luminosité
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(brightness)
            
            # Réduction de bruit
            if denoise:
                image = image.filter(ImageFilter.MedianFilter(size=3))
            
            return image
            
        except Exception as e:
            logger.warning(f"Erreur preprocessing image: {e}")
            # Retourne l'image originale en cas d'erreur
            return image
    
    def _smart_resize(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Redimensionne intelligemment en gardant le ratio d'aspect.
        
        Args:
            image: Image à redimensionner
            target_size: Taille cible (width, height)
            
        Returns:
            Image redimensionnée
        """
        original_width, original_height = image.size
        target_width, target_height = target_size
        
        # Calcul du ratio optimal
        width_ratio = target_width / original_width
        height_ratio = target_height / original_height
        ratio = min(width_ratio, height_ratio)
        
        # Nouvelle taille en gardant le ratio
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        
        # Redimensionnement
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Padding si nécessaire pour atteindre la taille exacte
        if (new_width, new_height) != target_size:
            # Création d'une image noire de la taille cible
            padded = Image.new('RGB', target_size, (0, 0, 0))
            
            # Centrage de l'image redimensionnée
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            padded.paste(resized, (x_offset, y_offset))
            
            return padded
        
        return resized
    
    def extract_keyframes(
        self,
        video_path: Union[str, Path],
        threshold: float = 0.3,
        max_frames: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Extrait les frames clés basées sur la différence entre images.
        
        Args:
            video_path: Chemin vers la vidéo
            threshold: Seuil de différence pour détecter un changement
            max_frames: Nombre max de frames
            
        Returns:
            Liste des frames clés
        """
        if max_frames is None:
            max_frames = settings.config.processing.max_frames
        
        path = Path(video_path)
        video_info = self.get_video_info(path)
        
        logger.info(f"Extraction keyframes avec seuil {threshold}")
        
        frames = []
        cap = None
        prev_frame = None
        
        try:
            cap = cv2.VideoCapture(str(path))
            
            success, image = cap.read()
            frame_count = 0
            
            while success and len(frames) < max_frames:
                # Conversion en niveaux de gris pour la comparaison
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Calcul de la différence avec la frame précédente
                    diff = cv2.absdiff(prev_frame, gray)
                    mean_diff = np.mean(diff) / 255.0  # Normalisation
                    
                    # Si la différence dépasse le seuil, c'est une keyframe
                    if mean_diff > threshold:
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(rgb_image)
                        processed_image = self.preprocess_frame(pil_image)
                        frames.append(processed_image)
                        
                        logger.debug(f"Keyframe détectée à {frame_count}: diff={mean_diff:.3f}")
                else:
                    # Première frame toujours incluse
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_image)
                    processed_image = self.preprocess_frame(pil_image)
                    frames.append(processed_image)
                
                prev_frame = gray.copy()
                success, image = cap.read()
                frame_count += 1
            
            logger.info(f"✅ {len(frames)} keyframes extraites")
            return frames
            
        except Exception as e:
            logger.error(f"❌ Erreur extraction keyframes: {e}")
            raise
        finally:
            if cap:
                cap.release()


# Instance globale du processeur vidéo
video_processor = VideoProcessor()

# Fonctions de compatibilité avec l'ancien code
def extract_frames(video_path: Union[str, Path], seconds_per_frame: float = 2.0) -> List[Image.Image]:
    """
    Fonction de compatibilité pour l'extraction de frames.
    
    Args:
        video_path: Chemin vers la vidéo
        seconds_per_frame: Intervalle entre frames
        
    Returns:
        Liste d'images PIL prétraitées
    """
    return video_processor.extract_frames(video_path, seconds_per_frame)


def preprocess_frame(
    pil_image: Image.Image,
    target_size: Tuple[int, int] = (384, 384),
    contrast: float = 1.5,
    brightness: float = 1.2,
    denoise: bool = True
) -> Image.Image:
    """
    Fonction de compatibilité pour le préprocessing.
    
    Args:
        pil_image: Image à traiter
        target_size: Taille cible
        contrast: Facteur de contraste
        brightness: Facteur de luminosité
        denoise: Activer réduction bruit
        
    Returns:
        Image prétraitée
    """
    return video_processor.preprocess_frame(
        pil_image, target_size, contrast, brightness, denoise
    )