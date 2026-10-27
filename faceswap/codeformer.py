"""Optional CodeFormer face restorer. Weights download on first use."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .utils import CODEFORMER_SHA256, CODEFORMER_URLS, MODELS_DIR, download_file, logger

# Published SHA256 of the official v0.1.0 codeformer.pth (359.2 MiB).
# Hugging Face LFS pointer for the same release asset:
# 1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7


def codeformer_install_tip() -> Optional[str]:
    """How to enable CodeFormer, or None when torch can be imported."""
    try:
        import torch  # noqa: F401
    except Exception:
        return (
            "CodeFormer needs PyTorch, which is not installed in this Python.\n\n"
            "On the Lenovo CUDA install, torch 2.x with cu124 is already present. "
            "The first CodeFormer swap downloads codeformer.pth into models/."
        )
    return None


def ensure_codeformer():
    target = MODELS_DIR / "codeformer.pth"
    return download_file(CODEFORMER_URLS, target, CODEFORMER_SHA256)


def load_codeformer(use_gpu: bool = True):
    """Build CodeFormer and load the official weights. Raises if torch is missing."""
    import torch

    from .codeformer_net import CodeFormer

    weight = ensure_codeformer()
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    model = CodeFormer()
    try:
        checkpoint = torch.load(weight, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(weight, map_location="cpu")
    state = checkpoint
    if isinstance(checkpoint, dict):
        state = checkpoint.get("params_ema") or checkpoint.get("params") or checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)
    logger.info("CodeFormer ready on %s", device)
    return model


def restore_bgr(model, bgr: np.ndarray, fidelity: float) -> np.ndarray:
    """Run CodeFormer on a BGR crop. ``fidelity`` 0 restores harder; 1 stays closer."""
    import cv2
    import torch

    if bgr is None or bgr.size == 0:
        return bgr
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = (tensor - 0.5) / 0.5
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    weight = float(np.clip(fidelity, 0.0, 1.0))
    with torch.inference_mode():
        output = model(tensor, w=weight, adain=True)[0]
    output = (output.clamp(-1, 1) + 1) / 2
    image = output.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    restored = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if restored.shape[:2] != bgr.shape[:2]:
        restored = cv2.resize(restored, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    return restored


def blend_restored(original: np.ndarray, restored: np.ndarray, strength: float) -> np.ndarray:
    """Mix the restorer output back onto the swap. 0 keeps the swap, 1 is fully restored."""
    import cv2

    amount = float(np.clip(strength, 0.0, 1.0))
    if amount <= 0.0 or restored is None:
        return original
    if restored.shape != original.shape:
        restored = cv2.resize(restored, (original.shape[1], original.shape[0]))
    if amount >= 0.999:
        return restored
    return cv2.addWeighted(restored, amount, original, 1.0 - amount, 0.0)
