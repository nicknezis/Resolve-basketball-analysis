"""Team classification from jersey crops with SigLIP embeddings.

Median HSV of a torso crop is fragile under gym lighting, log footage and
partially occluded torsos.  A vision-language embedding (SigLIP) of the same
crop separates "red pinnie" from "teal shirt" far more reliably, and does so
in a space where referees' stripes and spectators' clothing land far from
both team clusters, so they can be left unassigned instead of forced into a
team.

Pipeline per clip: up to N torso crops per track → SigLIP image embeddings
(batched on MPS/CUDA when available) → mean embedding per track → PCA to a
few dimensions → k-means (k=2) → tracks far from both centres stay ``None``.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_siglip(model_name: str, device: str):
    """Load the SigLIP model once per process (it is ~200M parameters)."""
    import torch
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    logger.info("Loaded %s on %s for team classification", model_name, device)
    return processor, model, device


class JerseyEmbedder:
    """Embeds BGR crops with SigLIP; returns L2-normalised vectors."""

    def __init__(self, model_name: str = "google/siglip-base-patch16-224", device: str = "auto", batch_size: int = 64):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

    def embed(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        import torch
        from PIL import Image

        processor, model, device = _load_siglip(self.model_name, self.device)
        out: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(crops_bgr), self.batch_size):
                batch = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr[i: i + self.batch_size]]
                inputs = processor(images=batch, return_tensors="pt").to(device)
                feats = model.get_image_features(**inputs)
                feats = getattr(feats, "pooler_output", feats)
                feats = torch.nn.functional.normalize(feats.float(), dim=-1)
                out.append(feats.cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1), dtype=np.float32)


def cluster_two_teams(
    features: np.ndarray,
    pca_dims: int = 3,
    outlier_std: float = 2.0,
) -> tuple[np.ndarray, dict]:
    """k-means (k=2) on PCA-reduced features; far-from-both-centres rows get -1.

    Returns ``(labels, info)`` with labels in {0, 1, -1}.
    """
    n = features.shape[0]
    if n < 2:
        return np.full(n, -1), {"reason": "too few tracks"}
    x = features.astype(np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    k = int(max(1, min(pca_dims, n - 1, x.shape[1])))
    # PCA via SVD
    _u, _s, vt = np.linalg.svd(x, full_matrices=False)
    z = (x @ vt[:k].T).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-4)
    compactness, labels, centers = cv2.kmeans(z, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel().astype(int)

    # Distance of each row to its own centre; flag outliers (refs, spectators)
    d = np.linalg.norm(z - centers[labels], axis=1)
    thresh = d.mean() + outlier_std * d.std() if n >= 6 else np.inf
    labels = np.where(d > thresh, -1, labels)

    sep = float(np.linalg.norm(centers[0] - centers[1]))
    spread = float(d[labels >= 0].mean()) if np.any(labels >= 0) else 0.0
    info = {
        "separation": sep,
        "mean_within": spread,
        "separation_ratio": sep / spread if spread > 0 else float("inf"),
        "outliers": int(np.sum(labels < 0)),
        "sizes": (int(np.sum(labels == 0)), int(np.sum(labels == 1))),
    }
    return labels, info
