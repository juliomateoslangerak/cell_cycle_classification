"""
Standalone cell cycle phase prediction.

Reproduces the inference pipeline from the predict_cell_cycle_phase notebook
without relying on cnn_framework or cell_cycle_classification.

Dependencies: torch, torchvision, tifffile, albumentations, numpy, huggingface_hub
"""

import json
import os

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
EDGE = 10
SCALE = 0.5

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


def _ensure_weights(models_dir: str) -> str:
    """Download pretrained weights from HuggingFace if not already present."""
    model_dir = os.path.join(models_dir, MODEL_SUBFOLDER)
    if not os.path.isdir(model_dir):
        print(f"Downloading pretrained weights from {HUGGINGFACE_REPO} ...")
        snapshot_download(HUGGINGFACE_REPO, local_dir=models_dir)
    return model_dir


def load_model(ccc_models_dir: str = "models", stardist_model: str = "2D_versatile_fluo") -> tuple[FucciClassifier, dict, StarDist2D]:
    """Load and return all models needed for inference.

    Downloads the cell cycle classifier weights from HuggingFace if not already
    present in `ccc_models_dir`.

    Parameters
    ----------
    ccc_models_dir : str
        Directory where the cell cycle classifier weights are (or will be) stored.
    stardist_model : str
        Name of the pretrained StarDist model used for nucleus segmentation.

    Returns
    -------
    ccc_model : FucciClassifier
        Cell cycle classifier in eval mode.
    ccc_mean_std : dict
        Per-channel normalisation statistics (keys "mean" and "std").
    stardist_model : StarDist2D
        StarDist segmentation model.
    """
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

def predict_nucleus(
    array: np.ndarray,
    ccc_model: FucciClassifier,
    ccc_mean_std: dict,
) -> str:
    """Predict the cell cycle phase for a single nucleus crop.

    Parameters
    ----------
    array : np.ndarray
        Shape (Z, H, W). uint16 or float32 in [0, 1].
    ccc_model : FucciClassifier
        Model returned by :func:`load_model`.
    ccc_mean_std : dict
        Normalisation statistics returned by :func:`load_model`.

    Returns
    -------
    str
        One of "G1", "S", or "G2/M".
    """
    pre_processed = preprocess(array, ccc_mean_std)
    with torch.no_grad():
        logits = ccc_model(pre_processed)  # (1, 3)
    return CYCLE_PHASES[logits.argmax(dim=1).item()]


def segment_nuclei(
    image: np.ndarray,
    stardist_model: StarDist2D,
    edge: int = 10,
    scale: float = 0.5,
    H_CROP: int = 100,
    W_CROP: int = 100,
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """Segment nuclei in a field-of-view image and return their crops and centres.

    Parameters
    ----------
    image : np.ndarray
        Shape (Z, H, W), uint16 or float32. StarDist segmentation runs on the
        max-projection across Z.
    stardist_model : StarDist2D
        StarDist model returned by :func:`load_model`.
    edge : int
        Margin in pixels added around each detected bounding box. Nuclei whose
        padded bounding box touches the image border are discarded.
    scale : float
        Rescaling factor passed to StarDist (default 0.5).
    H_CROP : int
        Minimum nucleus crop height (default 100).
    W_CROP : int
        Minimum nucleus crop width (default 100).

    Returns
    -------
    crops : list[np.ndarray]
        Per-nucleus arrays of shape (Z, H_crop, W_crop).
    centers : list[tuple[float, float]]
        (y, x) centroid coordinates for each returned nucleus.
    """
    # StarDist expects a 2D image — use the max-projection across z for segmentation
    image_2d = image.max(axis=0) if image.ndim == 3 else image
    _, details = stardist_model.predict_instances(csbdeep_normalize(image_2d), scale=scale)

    z, h, w = image.shape  # (Z, H, W)
    crops = []
    centers = []
    for coord in details["coord"]:
        y0, y1 = int(np.floor(coord[0].min())) - edge, int(np.ceil(coord[0].max())) + edge
        x0, x1 = int(np.floor(coord[1].min())) - edge, int(np.ceil(coord[1].max())) + edge
        # skip nuclei touching the image border
        if y0 <= 0 or y1 >= h or x0 <= 0 or x1 >= w:
            continue
        if y1 - y0 < H_CROP or x1 - x0 < W_CROP:
            continue
        crops.append(image[:, y0:y1, x0:x1])  # (Z, H_crop, W_crop)
        centers.append((coord[0].mean(), coord[1].mean()))

    return crops, centers


def predict_image(image: np.ndarray, stardist_model, ccc_model, ccc_mean_std, edge, scale):
    nuclei_crops, centers = segment_nuclei(
        image=image,
        stardist_model=stardist_model,
        edge=edge,
        scale=scale
    )

    phases = [predict_nucleus(crop, ccc_model, ccc_mean_std) for crop in nuclei_crops]

    return centers, phases

def predict_dir(
    directory: str,
    stardist_model: StarDist2D,
    ccc_model: FucciClassifier,
    ccc_mean_std: dict,
    edge: int = EDGE,
    scale: float = SCALE,
) -> dict[str, tuple[list[tuple[float, float]], list[str]]]:
    """Predict cell cycle phases for all TIFF images in a directory.

    Parameters
    ----------
    directory : str
        Path to the directory containing TIFF images.
    stardist_model : StarDist2D
        StarDist model returned by :func:`load_model`.
    ccc_model : FucciClassifier
        Cell cycle classifier returned by :func:`load_model`.
    ccc_mean_std : dict
        Normalisation statistics returned by :func:`load_model`.
    edge : int
        Margin in pixels added around each nucleus bounding box.
    scale : float
        Rescaling factor passed to StarDist.

    Returns
    -------
    dict[str, tuple[list[tuple[float, float]], list[str]]]
        Maps each filename to a (centers, phases) tuple.
    """
    results = {}
    tiff_files = [f for f in os.listdir(directory) if f.lower().endswith((".tif", ".tiff"))]

    for filename in tiff_files:
        image = tifffile.imread(os.path.join(directory, filename))
        if image.ndim == 2:
            image = np.stack([image] * IN_CHANNELS, axis=0)
        centers, phases = predict_image(image, stardist_model, ccc_model, ccc_mean_std, edge, scale)
        results[filename] = (centers, phases)

    return results


if __name__ == "__main__":
    ccc_model, ccc_mean_std, stardist_model = load_model()

    results = predict_dir("notebooks/Study_26", stardist_model, ccc_model, ccc_mean_std)

    for filename, (centers, phases) in results.items():
        print(f"\n{filename}:")
        for center, phase in zip(centers, phases):
            print(f"  {phase} at {center}")
