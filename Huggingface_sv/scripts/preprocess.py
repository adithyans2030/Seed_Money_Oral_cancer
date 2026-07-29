import logging
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms

logger = logging.getLogger("oralguard.preprocess")

def preprocess_image(img_path_or_pil, min_resolution=100):
    """
    Standardized preprocessing for OralGuard images (training and inference).
    Returns a torch tensor of shape (3, 224, 224) ready for MobileNetV2,
    or None if the image doesn't meet minimum resolution criteria.
    """
    # 1. Load image
    if isinstance(img_path_or_pil, str):
        try:
            img = Image.open(img_path_or_pil)
        except Exception as e:
            logger.warning(f"Failed to open image at {img_path_or_pil}: {e}")
            return None
    elif isinstance(img_path_or_pil, Image.Image):
        img = img_path_or_pil
    else:
        logger.warning("Input must be a file path string or PIL Image.")
        return None

    # 2. Reject if either dimension < min_resolution
    w, h = img.size
    if w < min_resolution or h < min_resolution:
        logger.warning(f"Image rejected: resolution {w}x{h} is below minimum {min_resolution}x{min_resolution}.")
        return None

    # 5. Convert to RGB (handle RGBA, grayscale)
    # We do this early so CLAHE has a consistent format (3 channels)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # 3. Apply CLAHE contrast enhancement
    # Convert PIL Image to OpenCV (numpy array)
    img_np = np.array(img)
    # PIL is RGB, OpenCV uses BGR, but we convert directly to LAB from RGB
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    img_clahe_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    img_clahe = Image.fromarray(img_clahe_np)

    # 4, 6, 7: Resize to 224x224 (LANCZOS), ToTensor, Normalize
    # We use torchvision transforms for this
    preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tensor = preprocess(img_clahe)
    return tensor
