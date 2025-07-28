# src/utils/preprocessing.py

from PIL import Image, ImageEnhance, ImageFilter

def preprocess_frame(pil_image, target_size=(384, 384), contrast=1.5, brightness=1.2, denoise=True):
    """
    Améliore la perception visuelle d'une image pour une meilleure analyse VLM.

    Args:
        pil_image (PIL.Image): Image originale en format PIL.
        target_size (tuple): Taille finale (largeur, hauteur).
        contrast (float): Facteur d'amélioration du contraste (1.0 = inchangé).
        brightness (float): Facteur de luminosité (1.0 = inchangé).
        denoise (bool): Appliquer ou non un filtre de réduction de bruit.

    Returns:
        PIL.Image: Image transformée.
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
