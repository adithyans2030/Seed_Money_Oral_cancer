import base64
from io import BytesIO
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

def get_gradcam_heatmap(model, target_layer, tensor_img, target_embedding):
    """
    Computes Grad-CAM for the cosine similarity between the image and target_embedding.
    Returns the heatmap as a numpy array.
    """
    model.eval()
    
    tensor_img = tensor_img.clone()
    tensor_img.requires_grad = True
    
    gradients = []
    activations = None
    
    # Iterate manually to avoid unreliable module hooks
    x = tensor_img
    for module in model.features:
        x = module(x)
        if module is target_layer:
            activations = x
            x.register_hook(lambda grad: gradients.append(grad))
            
    pooled = nn.AdaptiveAvgPool2d((1, 1))(x)
    query_emb = nn.Flatten()(pooled)
    query_emb_norm = F.normalize(query_emb, p=2, dim=1)
    
    target_emb_tensor = torch.tensor(target_embedding, dtype=torch.float32, device=query_emb_norm.device).unsqueeze(0)
    target_emb_norm = F.normalize(target_emb_tensor, p=2, dim=1)
    
    # Scalar to maximize: cosine similarity
    score = (query_emb_norm * target_emb_norm).sum()
    
    model.zero_grad()
    score.backward()
    
    if len(gradients) == 0 or activations is None:
        return None
        
    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations.detach().cpu().numpy()[0]
    
    weights = np.mean(grads, axis=(1, 2))
    
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]
        
    cam = np.maximum(cam, 0)
    if np.max(cam) > 0:
        cam = cam / np.max(cam)
    else:
        cam = np.zeros_like(cam)
        
    cam = cv2.resize(cam, (tensor_img.shape[3], tensor_img.shape[2]))
    return cam

def _guided_relu_backward_hook(module, grad_input, grad_output):
    """
    Guided backpropagation: in addition to the standard ReLU backward
    (zeroing gradient where the forward input was negative, which autograd
    already applies before this hook runs), also zero out negative
    gradients — only let positive influence flow backward. This is what
    turns plain backprop's noisy, everywhere-nonzero gradient into a clean,
    sparse, human-interpretable saliency map.
    """
    return (torch.clamp(grad_input[0], min=0.0),)


def guided_backprop_saliency(model, tensor_img, target_embedding):
    """
    Full-resolution (matches input size, no upsampling needed) saliency map
    for the same cosine-similarity score Grad-CAM explains, computed via
    guided backpropagation to the input image. Temporarily hooks every
    ReLU6 in the model (MobileNetV2 uses ReLU6 exclusively) and always
    removes the hooks afterward, since `model` is a shared object reused
    across requests — leaving them attached would corrupt normal inference.
    """
    relu_modules = [m for m in model.modules() if isinstance(m, nn.ReLU6)]
    # torchvision's MobileNetV2 uses inplace=True ReLU6, which conflicts with
    # backward hooks (autograd can't track a view that gets modified in place
    # inside a hooked custom Function) and raises a RuntimeError. Temporarily
    # switch to non-inplace for this pass only, restore after.
    original_inplace = [m.inplace for m in relu_modules]
    for m in relu_modules:
        m.inplace = False
    handles = [m.register_full_backward_hook(_guided_relu_backward_hook) for m in relu_modules]

    try:
        model.eval()
        tensor_img = tensor_img.clone().detach()
        tensor_img.requires_grad = True

        x = tensor_img
        for module in model.features:
            x = module(x)
        pooled = nn.AdaptiveAvgPool2d((1, 1))(x)
        query_emb = nn.Flatten()(pooled)
        query_emb_norm = F.normalize(query_emb, p=2, dim=1)

        target_emb_tensor = torch.tensor(target_embedding, dtype=torch.float32, device=query_emb_norm.device).unsqueeze(0)
        target_emb_norm = F.normalize(target_emb_tensor, p=2, dim=1)

        score = (query_emb_norm * target_emb_norm).sum()
        model.zero_grad()
        score.backward()

        if tensor_img.grad is None:
            return None

        grad = tensor_img.grad.data.cpu().numpy()[0]  # (3, H, W)
        saliency = np.max(np.abs(grad), axis=0)        # collapse channels
        if saliency.max() > 0:
            saliency = saliency / saliency.max()
        return saliency
    finally:
        for h in handles:
            h.remove()
        for m, was_inplace in zip(relu_modules, original_inplace):
            m.inplace = was_inplace


def get_guided_gradcam(model, target_layer, tensor_img, target_embedding):
    """
    Guided Grad-CAM: elementwise product of the coarse (7x7, upsampled)
    Grad-CAM localization map and a full-resolution guided-backprop
    saliency map. Grad-CAM alone is class-discriminative but blocky —
    the 7x7 feature grid can't represent fine detail. Guided backprop alone
    is full-resolution but not class-discriminative — it highlights every
    edge in the image regardless of relevance. Combining them keeps only
    the fine detail that falls inside the class-relevant coarse region.

    Two known failure modes of raw guided backprop, corrected below:
    - Backpropagating through 35 stacked ReLU6 layers (each clamping to
      positive-only) leaves only a handful of surviving pixels — visually
      near-invisible as isolated dots. Blurred into small visible patches.
    - What survives skews toward strong intensity edges anywhere in frame
      (e.g. a dark background / light skin boundary), not just the lesion.
      Raising the coarse Grad-CAM gate to a power > 1 before multiplying
      suppresses its weak/peripheral regions harder, concentrating the
      combined result toward the confident core of the class-relevant area.
    """
    cam = get_gradcam_heatmap(model, target_layer, tensor_img, target_embedding)
    if cam is None:
        return None
    saliency = guided_backprop_saliency(model, tensor_img, target_embedding)
    if saliency is None:
        return cam

    saliency = cv2.GaussianBlur(saliency.astype(np.float32), (15, 15), 0)
    if saliency.max() > 0:
        saliency = saliency / saliency.max()

    cam_gated = cam ** 1.5
    combined = cam_gated * saliency

    # Normalize against a high percentile rather than the absolute max, so
    # one extreme pixel doesn't wash out the rest of the visible signal.
    if combined.max() > 0:
        ref = np.percentile(combined, 99)
        if ref > 0:
            combined = np.clip(combined / ref, 0, 1)
    return combined


def generate_gradcam_base64(pil_img, heatmap):
    """
    Overlays the heatmap on the original PIL image and returns a Base64 string.
    """
    # Resize PIL image to 224x224 to match heatmap
    img_resized = pil_img.resize((224, 224))
    img_np = np.array(img_resized)
    
    # Ensure RGB
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
    heatmap_colored = np.uint8(255 * heatmap)
    colormap = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    
    superimposed_img = cv2.addWeighted(img_np, 0.5, colormap, 0.5, 0)
    
    # Convert to base64
    out_pil = Image.fromarray(superimposed_img)
    buffered = BytesIO()
    out_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return f"data:image/png;base64,{img_str}"
