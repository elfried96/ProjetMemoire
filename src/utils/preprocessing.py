# src/utils/preprocessing.py

import cv2
from PIL import Image, ImageEnhance, ImageFilter

def preprocess_frame(pil_image, target_size=(384, 384), contrast=1.5, brightness=1.2, denoise=True):
    """
    Améliore la perception visuelle d'une image pour une meilleure analyse VLM.
    """
    # Redimensionnement
    img = pil_image.resize(target_size)

    # Contraste
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)

    # Luminosité
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)

    # Réduction de bruit
    if denoise:
        img = img.filter(ImageFilter.MedianFilter(size=3))

    return img


def extract_frames(video_path, seconds_per_frame=2):
    """
    Extrait des images toutes les X secondes à partir d'une vidéo.

    Args:
        video_path (str): Chemin vers la vidéo.
        seconds_per_frame (int): Intervalle de temps entre deux frames extraites.

    Returns:
        list: Liste d'images PIL prétraitées.
    """
    frames = []
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * seconds_per_frame)

    success, image = vidcap.read()
    count = 0
    while success:
        if count % frame_interval == 0:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            processed = preprocess_frame(pil_image)
            frames.append(processed)
        success, image = vidcap.read()
        count += 1

    vidcap.release()
    print(f"✅ {len(frames)} frames extraites et prétraitées.")
    return frames
