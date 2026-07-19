"""
Standalone cell cycle phase prediction.

Reproduces the inference pipeline from the predict_cell_cycle_phase notebook
without relying on cnn_framework or cell_cycle_classification.

Dependencies: torch, torchvision, tifffile, albumentations, numpy, huggingface_hub
"""

import json
import os
import sys
from skimage.filters import threshold_otsu

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from stardist.models import StarDist2D
from csbdeep.utils import normalize as csbdeep_normalize
import tifffile
from huggingface_hub import snapshot_download
from torchvision.models import ResNet18_Weights, resnet18

# ── Constants ─────────────────────────────────────────────────────────────────

HUGGINGFACE_REPO = "thomas-bonte/cell_cycle_classification"
MODEL_SUBFOLDER = "20241101-055937-4998324"
MODEL_FILENAME = "early_stopping_cycle_classification.pt"
MEAN_STD_FILENAME = "mean_std.json"

IN_CHANNELS = 5       # 1 DAPI channel × 5 z-slices
LATENT_DIM = 256
NB_CLASSES = 3
CYCLE_PHASES = ["G1", "S", "G2/M"]

# Nucleus crop size (maximum nucleus diameter in pixels) and model input resolution
DATA_SET_SIZE = 280
INPUT_SIZE = 128


# ── Model architecture ─────────────────────────────────────────────────────────

def _redefine_first_layer(model, in_channels: int) -> None:
    """Replace ResNet's first conv to accept an arbitrary number of input channels.

    Extra channels beyond 3 are initialised with the mean of the ImageNet weights
    (same strategy as the original codebase).
    """
    orig = model.conv1.weight.data  # (64, 3, 7, 7)
    new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if in_channels >= 3:
        new_conv.weight.data[:, :3] = orig
        mean_weights = orig.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
        new_conv.weight.data[:, 3:] = mean_weights.expand(-1, in_channels - 3, -1, -1)
    else:
        mean_weights = orig.mean(dim=1, keepdim=True)
        new_conv.weight.data = mean_weights.expand(-1, in_channels, -1, -1)
    model.conv1 = new_conv


class _ResnetEncoder(nn.Module):
    """ResNet18 backbone with VAE-style linear heads.

    The state-dict keys match the original ResnetEncoder so that pretrained
    weights can be loaded directly with load_state_dict().
    """

    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        _redefine_first_layer(backbone, in_channels)
        feat_size = backbone.fc.in_features  # 512 for ResNet18
        backbone.fc = nn.Identity()
        self.conv_layers = backbone

        # These names must match the saved checkpoint exactly
        self.embedding = nn.Linear(feat_size, latent_dim)
        self.log_var = nn.Linear(feat_size, latent_dim)
        self.fucci = nn.Sequential(nn.Linear(latent_dim, 2), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_layers(x)           # (B, 512)
        return self.embedding(h)          # (B, latent_dim)


class FucciClassifier(nn.Module):
    """Frozen ResNet18 encoder followed by a two-layer MLP classifier."""

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        latent_dim: int = LATENT_DIM,
        nb_classes: int = NB_CLASSES,
    ):
        super().__init__()
        self.encoder = _ResnetEncoder(in_channels, latent_dim)
        self.fc1 = nn.Linear(latent_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, nb_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)             # (B, latent_dim)
        return self.fc2(self.fc1(emb))    # (B, nb_classes)


# ── Image preprocessing ────────────────────────────────────────────────────────

def preprocess(array: np.ndarray, mean_std: dict) -> torch.Tensor:
    """Preprocess a z-stack numpy array and return a model-ready tensor.

    Parameters
    ----------
    array : np.ndarray
        Shape (Z, H, W). uint16 values are divided by 65 535; float32 arrays
        are assumed to already be in [0, 1].
    mean_std : dict
        Normalisation statistics with keys "mean" and "std" (lists of length Z).

    Returns
    -------
    torch.Tensor of shape (1, Z, INPUT_SIZE, INPUT_SIZE)
    """
    if array.dtype == np.uint16:
        img = array.astype(np.float32) / 65535.0
    else:
        img = array.astype(np.float32)

    img = np.moveaxis(img, 0, -1)  # (H, W, Z) — albumentations expects HWC

    transform = A.Compose([
        A.Normalize(
            mean=mean_std["mean"],
            std=mean_std["std"],
            max_pixel_value=1.0,
            p=1.0,
        ),
        A.PadIfNeeded(
            min_height=DATA_SET_SIZE,
            min_width=DATA_SET_SIZE,
            border_mode=1,          # cv2.BORDER_REPLICATE
            p=1.0,
        ),
        A.CenterCrop(height=DATA_SET_SIZE, width=DATA_SET_SIZE, p=1.0),
        A.Resize(height=INPUT_SIZE, width=INPUT_SIZE, p=1.0),
    ])

    img = transform(image=img)["image"]  # (INPUT_SIZE, INPUT_SIZE, Z)
    img = np.clip(img, 0.0, 1.0)

    tensor = torch.from_numpy(img).permute(2, 0, 1).float()  # (Z, H, W)
    return tensor.unsqueeze(0)                                # (1, Z, H, W)


def load_and_preprocess(tiff_path: str, mean_std: dict) -> torch.Tensor:
    """Load a z-stack TIFF and return a model-ready tensor.

    Thin wrapper around :func:`preprocess` for TIFF files.
    Expected layout: (Z, H, W) uint16 — one DAPI channel, 5 z-slices.
    """
    return preprocess(tifffile.imread(tiff_path), mean_std)


# ── Model loading ──────────────────────────────────────────────────────────────

def _ensure_weights(models_dir: str) -> str:
    """Download pretrained weights from HuggingFace if not already present."""
    model_dir = os.path.join(models_dir, MODEL_SUBFOLDER)
    if not os.path.isdir(model_dir):
        print(f"Downloading pretrained weights from {HUGGINGFACE_REPO} ...")
        snapshot_download(HUGGINGFACE_REPO, local_dir=models_dir)
    return model_dir


def load_model(ccc_models_dir: str = "models", stardist_model: str = "2D_versatile_fluo") -> tuple[FucciClassifier, dict, StarDist2D]:
    """Return a ready-to-use (eval-mode) FucciClassifier and its mean/std dict."""
    ccc_model_dir = _ensure_weights(ccc_models_dir)
    ccc_model_path = os.path.join(ccc_model_dir, MODEL_FILENAME)
    ccc_mean_std_path = os.path.join(ccc_model_dir, MEAN_STD_FILENAME)

    with open(ccc_mean_std_path) as f:
        ccc_mean_std = json.load(f)

    ccc_model = FucciClassifier()
    state_dict = torch.load(ccc_model_path, map_location="cpu", weights_only=True)
    ccc_model.load_state_dict(state_dict)
    ccc_model.eval()

    stardist_model = StarDist2D.from_pretrained(stardist_model)

    return ccc_model, ccc_mean_std, stardist_model


# ── Inference ──────────────────────────────────────────────────────────────────

def predict_array(
    arrays: list[np.ndarray],
    ccc_model: FucciClassifier,
    ccc_mean_std: dict,
) -> list[str]:
    """Predict the cell cycle phase for a list of nucleus crop arrays.

    Parameters
    ----------
    arrays : list[np.ndarray]
        Each array has shape (Z, H, W). uint16 or float32 in [0, 1].
    ccc_model : FucciClassifier
        Model returned by :func:`load_model`.
    ccc_mean_std : dict
        Normalisation statistics returned by :func:`load_model`.

    Returns
    -------
    list[str]
        One of "G1", "S", or "G2/M" per input array.
    """
    batch = torch.cat([preprocess(a, ccc_mean_std) for a in arrays], dim=0)  # (N, Z, H, W)
    with torch.no_grad():
        logits = ccc_model(batch)                                              # (N, 3)
    return [CYCLE_PHASES[i] for i in logits.argmax(dim=1).tolist()]


def processing(
    image: np.ndarray,
    ccc_model: FucciClassifier,
    ccc_mean_std: dict,
    stardist_model: StarDist2D,
    scale: float = 0.5,
) -> list[str]:
    """Segment nuclei in a field-of-view image and predict their cell cycle phase.

    Parameters
    ----------
    image : np.ndarray
        (Z, H, W) raw DAPI image (uint16 or float32). StarDist segmentation
        uses the max-projection across Z.
    ccc_model : FucciClassifier
        Model returned by :func:`load_model`.
    ccc_mean_std : dict
        Normalisation statistics returned by :func:`load_model`.
    stardist_model : StarDist2D
        StarDist model returned by :func:`load_model`.
    scale : float
        Rescaling factor passed to StarDist (default 0.5).

    Returns
    -------
    list[str]
        One of "G1", "S", or "G2/M" per detected nucleus.
    """
    # StarDist expects a 2D image — use the max-projection across z for segmentation
    image_2d = image.max(axis=0) if image.ndim == 3 else image
    _, details = stardist_model.predict_instances(csbdeep_normalize(image_2d), scale=scale)

    z, h, w = image.shape  # (Z, H, W)
    crops = []
    for coord in details["coord"]:
        y0, y1 = int(np.floor(coord[0].min())), int(np.ceil(coord[0].max()))
        x0, x1 = int(np.floor(coord[1].min())), int(np.ceil(coord[1].max()))
        # skip nuclei touching the image border
        if y0 <= 1 or y1 >= h - 1 or x0 <= 1 or x1 >= w - 1:
            continue
        crops.append(image[:, y0:y1, x0:x1])  # (Z, H_crop, W_crop)

    if not crops:
        return []
    return predict_array(crops, ccc_model, ccc_mean_std)


def predict(tiff_path: str, models_dir: str = "models") -> str:
    """Predict the cell cycle phase for a single TIFF (convenience wrapper).

    Loads the model on every call — use :func:`load_model` + :func:`predict_array`
    directly if you need to run inference on many images.

    Parameters
    ----------
    tiff_path : str
        Path to a (Z, H, W) uint16 TIFF of the DAPI-stained nucleus.
    models_dir : str
        Directory where pretrained weights are (or will be) stored.

    Returns
    -------
    str
        One of "G1", "S", or "G2/M".
    """
    ccc_model, ccc_mean_std, stardist_model = load_model(models_dir)
    return predict_array([tifffile.imread(tiff_path)], ccc_model, ccc_mean_std)[0]


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    image = tifffile.imread("notebooks/Study_26/image_1702_Nucleus.ome.tiff")

    # If the image is 2D (H, W), tile it to (Z, H, W) to match the model's expected input
    if image.ndim == 2:
        image = np.stack([image] * IN_CHANNELS, axis=0)

    ccc_model, ccc_mean_std, stardist_model = load_model()

    phase = processing(image, ccc_model, ccc_mean_std, stardist_model)
    print(f"Predicted cell cycle phase: {phase}")
