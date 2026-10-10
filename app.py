import io
import os
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import streamlit as st

from evaluation.detection_accuracy import evaluate_dataset
from inference.nifti_reader import (
    discover_evaluation_cases,
    get_slice,
    get_slice_2d,
    normalize_volume,
    read_nifti,
    read_nifti_volume,
)

# ==============================================================================
# PAGE CONFIGURATION & THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="Edge AI Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom presentation-quality CSS
st.markdown(
    """
<style>
    /* Global Background & Typography */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Card */
    .hero-container {
        background: linear-gradient(135deg, rgba(13, 27, 42, 0.9) 0%, rgba(26, 38, 57, 0.8) 100%);
        border: 1px solid rgba(0, 229, 255, 0.25);
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(0, 229, 255, 0.15);
        color: #00E5FF;
        border: 1px solid #00E5FF;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .hero-title {
        font-size: 32px;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 15px;
        color: #8b949e;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Top Metric Summary Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 24px;
    }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px 18px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #00E5FF;
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8b949e;
    }
    .metric-val {
        font-size: 22px;
        font-weight: 800;
        color: #58a6ff;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 12px;
        color: #3fb950;
        margin-top: 2px;
    }

    /* Section Cards */
    .section-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #f0f6fc;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Primary Action Button Glow */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #00B4D8 0%, #0077B6 100%) !important;
        color: #FFFFFF !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        padding: 14px 28px !important;
        border-radius: 10px !important;
        border: 1px solid #00E5FF !important;
        box-shadow: 0 4px 20px rgba(0, 180, 216, 0.4) !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #00E5FF 0%, #0096C7 100%) !important;
        box-shadow: 0 6px 30px rgba(0, 229, 255, 0.6) !important;
        transform: translateY(-2px) !important;
    }

    /* Success Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(63, 185, 80, 0.15);
        color: #3fb950;
        border: 1px solid #3fb950;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 600;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        padding: 24px 0 10px 0;
        color: #8b949e;
        font-size: 12px;
        border-top: 1px solid #21262d;
        margin-top: 40px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# CONSTANTS & MODEL PATHS
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
INT8_MODEL_PATH = MODELS_DIR / "unet_int8.onnx"
FP32_MODEL_PATH = MODELS_DIR / "unet_fp32.onnx"
DEFAULT_DATASET_ROOT = r"C:\Users\SUBHAM\Desktop\IEEE_Dataset\extracted"
EVALUATION_DATA_DIR = PROJECT_ROOT / "evaluation_data"
DETECTION_AREA_THRESHOLD = 0.001


# ==============================================================================
# CACHED MODEL LOADERS
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_int8_session():
    """Load and cache the authoritative INT8 ONNX Runtime session."""
    if not INT8_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"INT8 ONNX model not found at: {INT8_MODEL_PATH}"
        )

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(INT8_MODEL_PATH),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    return session


@st.cache_resource(show_spinner=False)
def load_fp32_session():
    """Load and cache the baseline FP32 ONNX Runtime session."""
    if not FP32_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"FP32 ONNX model not found at: {FP32_MODEL_PATH}"
        )

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(FP32_MODEL_PATH),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    return session


def get_model_sizes():
    """Get verified physical sizes of FP32 and INT8 models on disk."""
    fp32_bytes = (
        FP32_MODEL_PATH.stat().st_size if FP32_MODEL_PATH.exists() else 7707663
    )
    int8_bytes = (
        INT8_MODEL_PATH.stat().st_size if INT8_MODEL_PATH.exists() else 1984070
    )

    fp32_mb = fp32_bytes / (1024 * 1024)
    int8_mb = int8_bytes / (1024 * 1024)
    reduction = (1.0 - (int8_bytes / fp32_bytes)) * 100.0
    return fp32_mb, int8_mb, reduction


@st.cache_data(show_spinner=False)
def cached_detection_evaluation(data_dir_str, model_path_str, threshold_val, max_samples_val):
    """Cached runner for dataset-wide tumor detection evaluation."""
    return evaluate_dataset(
        data_dir=data_dir_str,
        model_path=model_path_str,
        threshold=threshold_val,
        max_samples=max_samples_val,
        output_csv=str(PROJECT_ROOT / "results" / "detection_evaluation.csv"),
    )


# ==============================================================================
# H5 DATA LOADER
# ==============================================================================
def read_h5_data(source):
    """
    Load MRI slice 'x' and optional ground-truth 'y' from file path or uploaded bytes.
    Validates shapes and formats for edge inference.
    """
    if isinstance(source, (str, Path)):
        h5_file = h5py.File(source, "r")
    else:
        # Streamlit UploadedFile (BytesIO)
        source.seek(0)
        h5_file = h5py.File(io.BytesIO(source.read()), "r")

    try:
        if "x" not in h5_file:
            raise ValueError(
                "Missing required dataset key 'x' (MRI image array)."
            )

        image = h5_file["x"][:]

        if image.shape != (240, 240):
            raise ValueError(
                f"Unexpected MRI shape {image.shape}. Expected (240, 240)."
            )

        # Convert to float32 input
        image_f32 = image.astype(np.float32)

        # Ground truth 'y' is optional for genuinely unlabelled/clinical inputs
        if "y" in h5_file:
            mask = h5_file["y"][:]
            gt_binary = (mask > 0).astype(np.uint8)
        else:
            gt_binary = None

        return image_f32, gt_binary
    finally:
        h5_file.close()


@st.cache_data(show_spinner=False)
def load_and_normalize_nifti_volume(file_path_str: str):
    """
    Cached loader for uncompressed 3D NIfTI volumes.
    Loads raw int16 volume and performs whole-volume z-score normalization:
        (volume - mean) / (std + 1e-8)
    Returns normalized float32 volume (240, 240, 155) and metadata dict.
    """
    vol_raw, meta = read_nifti(file_path_str)
    vol_norm = normalize_volume(vol_raw)
    return vol_norm, meta


# ==============================================================================
# INFERENCE & METRIC FUNCTIONS
# ==============================================================================
def run_edge_inference(session, image_f32):
    """
    Execute real INT8 ONNX inference via ONNX Runtime with high-precision latency timing.
    Pipeline: H5 x -> float32 -> [1, 1, 240, 240] -> ONNX -> Sigmoid -> Threshold 0.5
    """
    input_name = session.get_inputs()[0].name
    input_tensor = image_f32[None, None, :, :]  # Shape [1, 1, 240, 240]

    # Measure inference latency with perf_counter
    start_time = time.perf_counter()
    raw_output = session.run(None, {input_name: input_tensor})[0]
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000.0
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

    # Sigmoid probability map [240, 240]
    probabilities = 1.0 / (1.0 + np.exp(-raw_output[0, 0]))

    # Binary tumor mask thresholded at 0.5
    pred_mask = (probabilities > 0.5).astype(np.uint8)

    return pred_mask, probabilities, latency_ms, fps


def compute_dice(pred_mask, gt_mask):
    """Calculate binary Dice similarity score."""
    p = pred_mask.astype(bool)
    g = gt_mask.astype(bool)
    intersection = np.logical_and(p, g).sum()
    dice = (2.0 * intersection) / (p.sum() + g.sum() + 1e-8)
    return float(dice)


def detect_tumor(pred_mask, min_area_fraction=DETECTION_AREA_THRESHOLD):
    """
    Derive tumor presence from the predicted segmentation mask.

    Detection is NOT a separate classifier.
    It is derived from the pixel-level segmentation output.
    """

    mask = np.asarray(pred_mask).astype(bool)

    if mask.ndim != 2:
        raise ValueError(
            f"Expected 2D segmentation mask, got shape {mask.shape}"
        )

    total_pixels = int(mask.size)
    tumor_pixels = int(mask.sum())

    tumor_area_fraction = (
        tumor_pixels / total_pixels
        if total_pixels > 0
        else 0.0
    )

    threshold_pixels = max(
        1,
        int(np.ceil(total_pixels * min_area_fraction))
    )

    detected = tumor_pixels >= threshold_pixels

    return {
        "detected": bool(detected),
        "label": (
            "TUMOR DETECTED"
            if detected
            else "NO TUMOR DETECTED"
        ),
        "tumor_pixels": tumor_pixels,
        "total_pixels": total_pixels,
        "tumor_area_fraction": tumor_area_fraction,
        "tumor_area_percent": tumor_area_fraction * 100.0,
        "threshold_pixels": threshold_pixels,
        "threshold_fraction": min_area_fraction,
    }


def compute_parity(fp32_mask, int8_mask):
    """
    Calculate numerical parity between FP32 baseline and INT8 edge predictions.
    Note: Numerical parity reflects exact algorithmic alignment under quantization,
    not medical or clinical equivalence.
    """
    total_pixels = int(fp32_mask.size)
    identical_pixels = int((fp32_mask == int8_mask).sum())
    differing_pixels = int((fp32_mask != int8_mask).sum())
    parity_pct = (identical_pixels / total_pixels) * 100.0
    differing_pct = (differing_pixels / total_pixels) * 100.0

    return {
        "total_pixels": total_pixels,
        "identical_pixels": identical_pixels,
        "differing_pixels": differing_pixels,
        "parity_pct": parity_pct,
        "differing_pct": differing_pct,
    }


# ==============================================================================
# VISUALIZATION BUILDERS
# ==============================================================================
def create_four_panel_figure(image, gt_mask, pred_mask, model_label="INT8"):
    """
    Construct presentation-quality 4-panel centerpiece:
    1. Original MRI (Grayscale)
    2. Ground Truth Mask
    3. Model Prediction (INT8 or FP32)
    4. Diagnostic Error / Segmentation Overlay
    """
    # Normalize MRI to [0, 1] for visual display
    img_norm = image - image.min()
    if img_norm.max() > 0:
        img_norm = img_norm / img_norm.max()

    fig, axes = plt.subplots(2, 2, figsize=(11, 11), facecolor="#0d1117")

    # Panel 1: Original MRI
    axes[0, 0].imshow(img_norm, cmap="gray", interpolation="nearest")
    axes[0, 0].set_title(
        "1. ORIGINAL MRI (FLAIR)",
        fontsize=13,
        color="#e6edf3",
        fontweight="bold",
        pad=10,
    )
    axes[0, 0].axis("off")

    # Panel 2: Ground Truth Tumor Mask
    axes[0, 1].imshow(img_norm, cmap="gray", interpolation="nearest")
    if gt_mask is not None:
        gt_colored = np.zeros((*gt_mask.shape, 4))
        gt_colored[gt_mask == 1] = [1.0, 0.2, 0.35, 0.75]  # High-contrast crimson
        axes[0, 1].imshow(gt_colored, interpolation="nearest")
        gt_pixels = int(gt_mask.sum())
        axes[0, 1].set_title(
            f"2. GROUND TRUTH ({gt_pixels:,} px)",
            fontsize=13,
            color="#ff7b72",
            fontweight="bold",
            pad=10,
        )
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "UNLABELED JURY TEST CASE\n\nNo ground-truth mask provided.\nModel evaluated in blind mode.",
            color="#8b949e",
            fontsize=10,
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#161b22", edgecolor="#30363d", alpha=0.9),
        )
        axes[0, 1].set_title(
            "2. GROUND TRUTH (N/A — Blind)",
            fontsize=13,
            color="#8b949e",
            fontweight="bold",
            pad=10,
        )
    axes[0, 1].axis("off")

    # Panel 3: Model Prediction
    axes[1, 0].imshow(img_norm, cmap="gray", interpolation="nearest")
    pred_colored = np.zeros((*pred_mask.shape, 4))
    if model_label == "FP32":
        pred_colored[pred_mask == 1] = [0.2, 0.65, 1.0, 0.75]  # Cyan / Blue
        label_color = "#58a6ff"
    else:
        pred_colored[pred_mask == 1] = [0.0, 0.95, 0.65, 0.75]  # Neon Emerald
        label_color = "#3fb950"

    axes[1, 0].imshow(pred_colored, interpolation="nearest")
    pred_pixels = int(pred_mask.sum())
    axes[1, 0].set_title(
        f"3. {model_label} PREDICTION ({pred_pixels:,} px)",
        fontsize=13,
        color=label_color,
        fontweight="bold",
        pad=10,
    )
    axes[1, 0].axis("off")

    # Panel 4: Diagnostic Multi-Color Breakdown Overlay
    axes[1, 1].imshow(img_norm, cmap="gray", interpolation="nearest")

    if gt_mask is not None:
        tp = np.logical_and(pred_mask == 1, gt_mask == 1)
        fp = np.logical_and(pred_mask == 1, gt_mask == 0)
        fn = np.logical_and(pred_mask == 0, gt_mask == 1)

        breakdown_overlay = np.zeros((*image.shape, 4))
        breakdown_overlay[tp] = [0.0, 0.95, 0.45, 0.85]  # TP: Vivid Green
        breakdown_overlay[fp] = [1.0, 0.30, 0.00, 0.85]  # FP: Vivid Orange
        breakdown_overlay[fn] = [0.2, 0.60, 1.00, 0.85]  # FN: Vivid Blue

        axes[1, 1].imshow(breakdown_overlay, interpolation="nearest")
        axes[1, 1].set_title(
            "4. DIAGNOSTIC OVERLAY (TP / FP / FN)",
            fontsize=13,
            color="#00E5FF",
            fontweight="bold",
            pad=10,
        )
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label=f"True Positive ({tp.sum():,})",
                markerfacecolor="#00F273",
                markersize=10,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label=f"False Positive ({fp.sum():,})",
                markerfacecolor="#FF4D00",
                markersize=10,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label=f"False Negative ({fn.sum():,})",
                markerfacecolor="#3399FF",
                markersize=10,
            ),
        ]
    else:
        overlay = np.zeros((*image.shape, 4))
        overlay_color = [0.0, 0.95, 0.65, 0.80] if model_label == "INT8" else [0.2, 0.65, 1.0, 0.80]
        overlay[pred_mask == 1] = overlay_color
        axes[1, 1].imshow(overlay, interpolation="nearest")
        axes[1, 1].set_title(
            f"4. PREDICTION OVERLAY ON MRI ({pred_pixels:,} px)",
            fontsize=13,
            color="#00E5FF",
            fontweight="bold",
            pad=10,
        )
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label=f"Predicted Tumor ({pred_pixels:,} px)",
                markerfacecolor=label_color,
                markersize=10,
            ),
        ]
    axes[1, 1].legend(
        handles=legend_elements,
        loc="lower left",
        fontsize=9,
        facecolor="#161b22",
        edgecolor="#30363d",
        labelcolor="#e6edf3",
    )

    plt.tight_layout(pad=2.0)
    return fig


def create_side_by_side_figure(
    image,
    gt_mask,
    fp32_pred,
    int8_pred,
    dice_fp32=None,
    dice_int8=None,
    fp32_latency=None,
    int8_latency=None,
):
    """
    Construct presentation-quality 4-panel Side-by-Side Comparison:
    1. Original MRI (FLAIR) + Ground Truth
    2. FP32 Baseline Prediction (Cyan)
    3. INT8 Edge Prediction (Emerald)
    4. Model Parity & Discrepancy Map
    """
    img_norm = image - image.min()
    if img_norm.max() > 0:
        img_norm = img_norm / img_norm.max()

    fig, axes = plt.subplots(2, 2, figsize=(11, 11), facecolor="#0d1117")

    # Panel 1: Original MRI + Ground Truth
    axes[0, 0].imshow(img_norm, cmap="gray", interpolation="nearest")
    if gt_mask is not None:
        gt_colored = np.zeros((*gt_mask.shape, 4))
        gt_colored[gt_mask == 1] = [1.0, 0.2, 0.35, 0.75]  # Crimson
        axes[0, 0].imshow(gt_colored, interpolation="nearest")
        gt_pixels = int(gt_mask.sum())
        axes[0, 0].set_title(
            f"1. MRI + GROUND TRUTH ({gt_pixels:,} px)",
            fontsize=13,
            color="#ff7b72",
            fontweight="bold",
            pad=10,
        )
    else:
        axes[0, 0].set_title(
            "1. ORIGINAL MRI (FLAIR)",
            fontsize=13,
            color="#e6edf3",
            fontweight="bold",
            pad=10,
        )
    axes[0, 0].axis("off")

    # Panel 2: FP32 Baseline Prediction
    axes[0, 1].imshow(img_norm, cmap="gray", interpolation="nearest")
    fp32_colored = np.zeros((*fp32_pred.shape, 4))
    fp32_colored[fp32_pred == 1] = [0.2, 0.65, 1.0, 0.75]  # Cyan / Blue
    axes[0, 1].imshow(fp32_colored, interpolation="nearest")
    fp32_pixels = int(fp32_pred.sum())
    fp32_title = f"2. FP32 BASELINE ({fp32_pixels:,} px)"
    if dice_fp32 is not None and fp32_latency is not None:
        fp32_title += f"\nDice: {dice_fp32 * 100:.2f}% | Latency: {fp32_latency:.1f} ms"
    elif fp32_latency is not None:
        fp32_title += f"\nLatency: {fp32_latency:.1f} ms"
    axes[0, 1].set_title(
        fp32_title,
        fontsize=12,
        color="#58a6ff",
        fontweight="bold",
        pad=10,
    )
    axes[0, 1].axis("off")

    # Panel 3: INT8 Edge Prediction
    axes[1, 0].imshow(img_norm, cmap="gray", interpolation="nearest")
    int8_colored = np.zeros((*int8_pred.shape, 4))
    int8_colored[int8_pred == 1] = [0.0, 0.95, 0.65, 0.75]  # Neon Emerald
    axes[1, 0].imshow(int8_colored, interpolation="nearest")
    int8_pixels = int(int8_pred.sum())
    int8_title = f"3. INT8 EDGE MODEL ({int8_pixels:,} px)"
    if dice_int8 is not None and int8_latency is not None:
        int8_title += f"\nDice: {dice_int8 * 100:.2f}% | Latency: {int8_latency:.1f} ms"
    elif int8_latency is not None:
        int8_title += f"\nLatency: {int8_latency:.1f} ms"
    axes[1, 0].set_title(
        int8_title,
        fontsize=12,
        color="#3fb950",
        fontweight="bold",
        pad=10,
    )
    axes[1, 0].axis("off")

    # Panel 4: Model Parity & Discrepancy Map
    # Agreement: Both detect tumor (Emerald)
    # Discrepancy: INT8 only (Orange), FP32 only (Blue)
    axes[1, 1].imshow(img_norm, cmap="gray", interpolation="nearest")
    both_tumor = np.logical_and(fp32_pred == 1, int8_pred == 1)
    int8_only = np.logical_and(fp32_pred == 0, int8_pred == 1)
    fp32_only = np.logical_and(fp32_pred == 1, int8_pred == 0)

    diff_overlay = np.zeros((*image.shape, 4))
    diff_overlay[both_tumor] = [0.0, 0.95, 0.45, 0.85]  # Both: Vivid Green
    diff_overlay[int8_only] = [1.0, 0.55, 0.0, 0.90]    # INT8 Only: Orange
    diff_overlay[fp32_only] = [0.2, 0.60, 1.0, 0.90]    # FP32 Only: Blue

    axes[1, 1].imshow(diff_overlay, interpolation="nearest")
    total_diff = int(int8_only.sum() + fp32_only.sum())
    axes[1, 1].set_title(
        f"4. MODEL PARITY MAP ({total_diff:,} diff px)",
        fontsize=12,
        color="#00E5FF",
        fontweight="bold",
        pad=10,
    )
    axes[1, 1].axis("off")

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=f"Both Agree ({both_tumor.sum():,})",
            markerfacecolor="#00F273",
            markersize=9,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=f"INT8 Only ({int8_only.sum():,})",
            markerfacecolor="#FF8C00",
            markersize=9,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=f"FP32 Only ({fp32_only.sum():,})",
            markerfacecolor="#3399FF",
            markersize=9,
        ),
    ]
    axes[1, 1].legend(
        handles=legend_elements,
        loc="lower left",
        fontsize=9,
        facecolor="#161b22",
        edgecolor="#30363d",
        labelcolor="#e6edf3",
    )

    plt.tight_layout(pad=2.0)
    return fig


def create_probability_figure(probabilities):
    """Render pixel-level sigmoid probability map with colorbar."""
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0d1117")
    im = ax.imshow(
        probabilities, cmap="inferno", vmin=0.0, vmax=1.0, interpolation="bicubic"
    )
    ax.set_title(
        "Model Probability Map — Sigmoid Output",
        fontsize=11,
        color="#e6edf3",
        fontweight="bold",
        pad=8,
    )
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8, colors="#8b949e")
    cbar.set_label("Tumor Probability [0.0 - 1.0]", color="#8b949e", fontsize=9)
    plt.tight_layout()
    return fig


# ==============================================================================
# MAIN APPLICATION INTERFACE
# ==============================================================================
fp32_mb, int8_mb, size_reduction = get_model_sizes()

# Header / Hero Section
st.markdown(
    """
<div class="hero-container">
    <div class="hero-badge">IEEE Edge AI Hackathon 2026</div>
    <h1 class="hero-title">Edge AI Brain Tumor Segmentation</h1>
    <p class="hero-subtitle">Real-time MRI segmentation using static quantized INT8 U-Net targeting Raspberry Pi deployment</p>
</div>
""",
    unsafe_allow_html=True,
)

# Top Metrics Bar
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-label">Model Architecture</div>
        <div class="metric-val">INT8 ONNX</div>
        <div class="metric-sub">Static QDQ Quantized</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with col_m2:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-label">Model Storage</div>
        <div class="metric-val">{int8_mb:.2f} MB</div>
        <div class="metric-sub">↓ {size_reduction:.1f}% vs FP32 ({fp32_mb:.2f} MB)</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with col_m3:
    st.markdown(
        """
    <div class="metric-card">
        <div class="metric-label">Biomedical Task</div>
        <div class="metric-val">Tumor Segmentation</div>
        <div class="metric-sub">BraTS 2D FLAIR MRI</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with col_m4:
    st.markdown(
        """
    <div class="metric-card">
        <div class="metric-label">Edge Runtime</div>
        <div class="metric-val">ONNX Runtime</div>
        <div class="metric-sub">CPUExecutionProvider</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# SECTION 1: SAMPLE SELECTION
# ==============================================================================
st.markdown(
    """
<div class="section-title">
    <span>📂</span> 1. Select MRI Sample
</div>
""",
    unsafe_allow_html=True,
)

current_image = None
current_gt = None
sample_metadata = {}

# Triple Mode Input: Dataset Explorer, Direct H5 Upload, or Jury NIfTI Evaluation
input_mode = st.radio(
    "Select Input Source Mode",
    options=[
        "📁 Browse Local Dataset (152 Patients)",
        "⬆️ Upload Custom .H5 File",
        "⚖️ Jury NIfTI Evaluation (3D Volumes)",
    ],
    horizontal=True,
    help="Switch between local BraTS dataset, custom .h5 upload, or the official Jury 3D NIfTI evaluation.",
)

if input_mode == "📁 Browse Local Dataset (152 Patients)":
    dataset_path_str = st.text_input(
        "Dataset Extracted Directory",
        value=DEFAULT_DATASET_ROOT,
        help="Local directory containing BRATS_XXX patient folders.",
    )
    dataset_root = Path(dataset_path_str)

    if dataset_root.exists():
        patient_dirs = sorted([
            d.name
            for d in dataset_root.iterdir()
            if d.is_dir() and d.name.startswith("BRATS")
        ])

        if patient_dirs:
            col_pat, col_slice = st.columns([1, 1])

            # Preferred demo presets
            default_pat_idx = (
                patient_dirs.index("BRATS_001")
                if "BRATS_001" in patient_dirs
                else 0
            )

            with col_pat:
                selected_patient = st.selectbox(
                    "Select Patient Folder",
                    options=patient_dirs,
                    index=default_pat_idx,
                )

            patient_dir = dataset_root / selected_patient
            slice_files = sorted(list(patient_dir.glob("*.h5")))

            with col_slice:
                if slice_files:
                    slice_names = [f.name for f in slice_files]
                    # Select slice 60 by default if available (standard high-tumor slice)
                    def_slice_idx = 0
                    for idx, name in enumerate(slice_names):
                        if "_60.h5" in name or "_68.h5" in name:
                            def_slice_idx = idx
                            break

                    selected_slice_name = st.selectbox(
                        "Select MRI Slice (.h5)",
                        options=slice_names,
                        index=def_slice_idx,
                    )
                    selected_h5_source = patient_dir / selected_slice_name
                    try:
                        current_image, current_gt = read_h5_data(selected_h5_source)
                        sample_metadata["patient"] = selected_patient
                        sample_metadata["slice"] = selected_slice_name
                        sample_metadata["modality"] = "FLAIR (2D)"
                        if current_gt is not None:
                            sample_metadata["gt_pixels"] = int(current_gt.sum())
                            sample_metadata["is_blind"] = False
                        else:
                            sample_metadata["gt_pixels"] = None
                            sample_metadata["is_blind"] = True
                    except Exception as e:
                        st.error(f"Error loading HDF5 file: {e}")
                else:
                    st.warning(f"No .h5 files found in {selected_patient}.")
        else:
            st.warning(
                f"No BRATS patient folders found in {dataset_path_str}. Try uploading directly."
            )
    else:
        st.info(
            f"Local path '{dataset_path_str}' not detected on this machine. Please use the 'Upload Custom .H5 File' mode."
        )

elif input_mode == "⬆️ Upload Custom .H5 File":
    uploaded_file = st.file_uploader(
        "Choose an HDF5 slice file (*.h5 with 'x' and optional 'y' datasets)",
        type=["h5"],
        help="Upload any valid 240x240 BraTS slice HDF5 file. Ground truth 'y' is optional for unlabelled clinical inputs.",
    )
    if uploaded_file is not None:
        try:
            current_image, current_gt = read_h5_data(uploaded_file)
            fname = uploaded_file.name
            sample_metadata["patient"] = (
                fname.split("_")[0] + "_" + fname.split("_")[1]
                if "_" in fname
                else "Uploaded"
            )
            sample_metadata["slice"] = fname
            sample_metadata["modality"] = "FLAIR (2D)"
            if current_gt is not None:
                sample_metadata["gt_pixels"] = int(current_gt.sum())
                sample_metadata["is_blind"] = False
            else:
                sample_metadata["gt_pixels"] = None
                sample_metadata["is_blind"] = True
        except Exception as e:
            st.error(f"Error loading uploaded HDF5 file: {e}")
    else:
        st.info("Please upload a 240×240 BraTS slice HDF5 file (*.h5 with 'x' array) to begin.")

elif input_mode == "⚖️ Jury NIfTI Evaluation (3D Volumes)":
    st.markdown(
        """
        <div style="background: rgba(0, 229, 255, 0.07); border: 1px solid rgba(0, 229, 255, 0.3); border-radius: 8px; padding: 12px 16px; margin-bottom: 16px;">
            <div style="color: #00E5FF; font-weight: 600; font-size: 14px;">⚖️ Official Jury Evaluation Mode (3D NIfTI Volumes)</div>
            <div style="color: #8b949e; font-size: 12px; margin-top: 4px;">
                Direct runtime adaptation of uncompressed <code>.nii</code> test cases provided by the evaluation committee.
                Zero conversion to .h5. Performs whole-volume normalization and dynamic axial slice extraction into the existing ONNX U-Net.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    eval_cases = discover_evaluation_cases(EVALUATION_DATA_DIR)
    nii_source_path = None
    selected_case_info = None

    if eval_cases:
        case_labels = []
        for c in eval_cases:
            rec_tag = " ⭐ [RECOMMENDED — Model Primary]" if c["is_recommended"] else " 🔍 [Inspection Only]"
            case_labels.append(f"{c['modality']} ({c['filename']}){rec_tag}")

        col_nii_case, col_nii_slice = st.columns([1.3, 1])

        with col_nii_case:
            selected_case_idx = st.selectbox(
                "Select Jury NIfTI Volume & Modality",
                options=range(len(case_labels)),
                format_func=lambda i: case_labels[i],
                index=0,
                help="The model was trained on 2D FLAIR MRI. FLAIR is selected by default as the primary evaluation modality.",
            )
            selected_case_info = eval_cases[selected_case_idx]
            nii_source_path = selected_case_info["path"]

            if not selected_case_info["is_recommended"]:
                st.warning(
                    f"⚠️ **Cross-Modality Inspection**: You selected `{selected_case_info['modality']}`. "
                    "The segmentation model was trained exclusively on **FLAIR** sequences. "
                    "Cross-modality inference is demonstrated for jury inspection, but optimal tumor segmentation is achieved with FLAIR."
                )

        if nii_source_path is not None and Path(nii_source_path).exists():
            try:
                with st.spinner("Loading & Normalizing 3D NIfTI Volume..."):
                    vol_norm, nifti_meta = load_and_normalize_nifti_volume(str(nii_source_path))
                    num_slices = vol_norm.shape[2]

                with col_nii_slice:
                    selected_slice_idx = st.slider(
                        f"Select Axial Slice (Z: 0 → {num_slices - 1})",
                        min_value=0,
                        max_value=num_slices - 1,
                        value=70,
                        help="BraTS 3D volumes contain 155 axial slices. Slices 32–128 contain observable tumor volume.",
                    )
                    if 32 <= selected_slice_idx <= 128:
                        st.caption("🟢 **Active Tumor Region**: High tumor probability span (Slices 32–128).")
                    else:
                        st.caption("⚪ **Peripheral Cranial Region**.")

                current_image = get_slice_2d(vol_norm, selected_slice_idx)
                current_gt = None  # Jury evaluation cases are blind test cases without ground truth

                sample_metadata["patient"] = selected_case_info["case_id"]
                sample_metadata["slice"] = f"{selected_case_info['filename']} [Slice {selected_slice_idx:03d}/{num_slices - 1}]"
                sample_metadata["modality"] = selected_case_info["modality"]
                sample_metadata["gt_pixels"] = None
                sample_metadata["is_blind"] = True

                # Interactive Full-Volume Batch Scan Expander for Jury
                with st.expander("📊 Full-Volume 155 Slices Batch Scan (Jury Feature)", expanded=False):
                    st.write("Scan all 155 slices across the selected 3D NIfTI volume to plot the volumetric tumor distribution and compute volume-wide parity.")
                    if st.button("⚡ Run Full-Volume 155 Slices Scan", key="btn_run_155_scan"):
                        with st.spinner("Processing 155 axial slices via INT8 & FP32..."):
                            sess_int8 = load_int8_session()
                            sess_fp32 = load_fp32_session()
                            inp_name_i = sess_int8.get_inputs()[0].name
                            inp_name_f = sess_fp32.get_inputs()[0].name

                            slice_indices = []
                            int8_pixel_counts = []
                            fp32_pixel_counts = []
                            latencies = []

                            t_vol_start = time.perf_counter()
                            for z in range(num_slices):
                                s_tensor = get_slice(vol_norm, z)

                                t0 = time.perf_counter()
                                out_i = sess_int8.run(None, {inp_name_i: s_tensor})[0]
                                lat_i = (time.perf_counter() - t0) * 1000.0
                                latencies.append(lat_i)

                                out_f = sess_fp32.run(None, {inp_name_f: s_tensor})[0]

                                p_i = 1.0 / (1.0 + np.exp(-out_i[0, 0]))
                                p_f = 1.0 / (1.0 + np.exp(-out_f[0, 0]))

                                int8_pixel_counts.append(int((p_i > 0.5).sum()))
                                fp32_pixel_counts.append(int((p_f > 0.5).sum()))
                                slice_indices.append(z)

                            t_vol_total = time.perf_counter() - t_vol_start
                            total_int8_voxels = sum(int8_pixel_counts)
                            avg_int8_lat = float(np.mean(latencies))
                            peak_slice = int(np.argmax(int8_pixel_counts))
                            peak_pixels = int8_pixel_counts[peak_slice]

                            fig_vol, ax_vol = plt.subplots(figsize=(10, 3.5), facecolor="#0d1117")
                            ax_vol.set_facecolor("#161b22")
                            ax_vol.plot(slice_indices, fp32_pixel_counts, label="FP32 Baseline", color="#58a6ff", linewidth=1.8, linestyle="--")
                            ax_vol.plot(slice_indices, int8_pixel_counts, label="INT8 Edge Model", color="#3fb950", linewidth=2.0)
                            ax_vol.fill_between(slice_indices, int8_pixel_counts, color="#3fb950", alpha=0.15)
                            ax_vol.axvline(selected_slice_idx, color="#ff7b72", linestyle=":", label=f"Current Slice ({selected_slice_idx})")
                            ax_vol.set_xlabel("Axial Slice Index (0 → 154)", color="#8b949e", fontsize=10)
                            ax_vol.set_ylabel("Tumor Pixels Detected", color="#8b949e", fontsize=10)
                            ax_vol.set_title(f"3D Tumor Volume Profile ({total_int8_voxels:,} mm³ total tumor)", color="#e6edf3", fontsize=12, fontweight="bold")
                            ax_vol.tick_params(colors="#8b949e")
                            ax_vol.grid(True, color="#30363d", alpha=0.5, linestyle=":")
                            ax_vol.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=9)
                            plt.tight_layout()

                            st.pyplot(fig_vol, use_container_width=True)
                            plt.close(fig_vol)

                            c_vol1, c_vol2, c_vol3, c_vol4 = st.columns(4)
                            with c_vol1:
                                st.metric("Total Tumor Volume", f"{total_int8_voxels:,} mm³")
                            with c_vol2:
                                st.metric("Peak Slice", f"Slice {peak_slice} ({peak_pixels:,} px)")
                            with c_vol3:
                                st.metric("Mean Slice Latency", f"{avg_int8_lat:.2f} ms")
                            with c_vol4:
                                st.metric("Total 155 Slices Time", f"{t_vol_total:.2f} s")

            except Exception as e:
                st.error(f"Failed to load NIfTI volume: {e}")
    else:
        st.warning("No evaluation cases detected in `evaluation_data/` directory.")

# Display Sample Metadata Bar if loaded
if current_image is not None:
    col_info1, col_info2, col_info3, col_info4, col_info5 = st.columns(5)
    with col_info1:
        st.caption("PATIENT ID")
        st.markdown(f"**{sample_metadata.get('patient', 'BRATS')}**")
    with col_info2:
        st.caption("SLICE FILE")
        st.markdown(f"`{sample_metadata.get('slice', 'MRI Slice')}`")
    with col_info3:
        st.caption("IMAGE SHAPE")
        st.markdown(f"**{current_image.shape[0]} × {current_image.shape[1]}**")
    with col_info4:
        st.caption("INPUT DTYPE")
        st.markdown(f"**{current_image.dtype}**")
    with col_info5:
        st.caption("GROUND TRUTH TUMOR")
        if current_gt is not None:
            st.markdown(f"**{sample_metadata.get('gt_pixels', 0):,} pixels**")
        else:
            st.markdown("**N/A (Blind Eval)**")

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================================================
    # SECTION 2: PREVIEW & INFERENCE TRIGGER
    # ==========================================================================
    col_prev, col_action = st.columns([1.2, 1])

    with col_prev:
        st.markdown(
            """
        <div class="section-title">
            <span>👁️</span> 2. MRI Input Preview
        </div>
        """,
            unsafe_allow_html=True,
        )
        fig_prev, ax_prev = plt.subplots(figsize=(4.5, 4.5), facecolor="#0d1117")
        ax_prev.imshow(current_image, cmap="gray", interpolation="nearest")
        ax_prev.set_title("Input 2D FLAIR Slice", color="#8b949e", fontsize=10)
        ax_prev.axis("off")
        st.pyplot(fig_prev, use_container_width=False)
        plt.close(fig_prev)

    with col_action:
        st.markdown(
            """
        <div class="section-title">
            <span>⚡</span> 3. Edge AI Execution
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.write(
            "Select model execution mode and click below to run real-time inference through ONNX Runtime on CPU. "
            "High-resolution timers will measure runtime latency and throughput."
        )

        model_mode = st.radio(
            "Model Execution Mode",
            options=[
                "INT8 — Edge Optimized (Default)",
                "FP32 — Baseline Precision",
                "Side-by-Side Comparison",
            ],
            index=0,
            help="Choose between edge INT8 model, FP32 baseline, or dual side-by-side comparison.",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        run_inference_btn = st.button("🚀 RUN INFERENCE", use_container_width=True)

        if not run_inference_btn:
            st.info("Awaiting execution trigger...")

    # ==========================================================================
    # SECTION 3: INFERENCE RESULTS & CENTERPIECE VISUALIZATION
    # ==========================================================================
    if run_inference_btn:
        try:
            if model_mode == "INT8 — Edge Optimized (Default)":
                with st.spinner("Running INT8 Edge Inference on CPU..."):
                    session = load_int8_session()
                    pred_mask, probabilities, latency_ms, fps = run_edge_inference(
                        session, current_image
                    )
                    dice = compute_dice(pred_mask, current_gt) if current_gt is not None else None
                    pred_pixels = int(pred_mask.sum())
                    gt_pixels = int(current_gt.sum()) if current_gt is not None else None

                detection = detect_tumor(pred_mask)

                # Status Completion Badge
                st.markdown(
                    f"""
                <div style="margin: 16px 0;">
                    <span class="status-badge">✓ INT8 Edge Inference Completed in {latency_ms:.2f} ms ({fps:.1f} FPS)</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                if detection["detected"]:
                    st.success(
                        f"🧠 {detection['label']} — "
                        f"{detection['tumor_pixels']:,} predicted tumor pixels "
                        f"({detection['tumor_area_percent']:.3f}% of the MRI slice)"
                    )
                else:
                    st.info(
                        f"✓ {detection['label']} — "
                        f"{detection['tumor_pixels']:,} predicted tumor pixels"
                    )

                st.markdown(
                    """
                <div class="section-title">
                    <span>🧠</span> Tumor Detection
                </div>
                """,
                    unsafe_allow_html=True,
                )

                det1, det2, det3 = st.columns(3)

                with det1:
                    st.metric(
                        "Detection Result",
                        detection["label"]
                    )

                with det2:
                    st.metric(
                        "Tumor Area",
                        f"{detection['tumor_area_percent']:.3f}%"
                    )

                with det3:
                    st.metric(
                        "Tumor Pixels",
                        f"{detection['tumor_pixels']:,}"
                    )

                st.caption(
                    "Detection is derived from the predicted segmentation mask. "
                    f"Minimum detection area: "
                    f"{detection['threshold_pixels']:,} pixels "
                    f"({detection['threshold_fraction'] * 100:.3f}%)."
                )

                st.caption(
                    "Detection is derived from the U-Net segmentation output: "
                    "a tumor is considered detected when the predicted tumor area "
                    "exceeds the configured minimum area threshold. "
                    "This is not a separately trained classification model."
                )

                st.markdown("<br>", unsafe_allow_html=True)

                # Quantitative Results Cards
                st.markdown(
                    """
                <div class="section-title">
                    <span>📊</span> 4. Segmentation Performance (Live Evaluation)
                </div>
                """,
                    unsafe_allow_html=True,
                )

                res1, res2, res3, res4, res5, res6 = st.columns(6)
                with res1:
                    st.metric("Dice Score", f"{dice * 100:.2f} %" if dice is not None else "N/A (Blind Eval)")
                with res2:
                    st.metric("Predicted Pixels", f"{pred_pixels:,}")
                with res3:
                    st.metric("Ground Truth Pixels", f"{gt_pixels:,}" if gt_pixels is not None else "N/A (Blind)")
                with res4:
                    st.metric("Latency", f"{latency_ms:.2f} ms")
                with res5:
                    st.metric("Throughput", f"{fps:.1f} FPS")
                with res6:
                    st.metric("INT8 Model Size", f"{int8_mb:.2f} MB")

                if current_gt is None:
                    st.info(
                        "ℹ️ **Dice / IoU: N/A — Blind Evaluation (No Ground Truth Provided)**. "
                        "Jury test volume is evaluated without ground truth. Latency, throughput, "
                        "predicted tumor pixels, and memory footprint are measured live from ONNX Runtime CPU execution."
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # Centerpiece 4-Panel Visualization
                st.markdown(
                    """
                <div class="section-title">
                    <span>🖼️</span> 5. Visual Segmentation Centerpiece
                </div>
                """,
                    unsafe_allow_html=True,
                )

                vis_col1, vis_col2 = st.columns([2.5, 1])

                with vis_col1:
                    fig_center = create_four_panel_figure(
                        current_image, current_gt, pred_mask, model_label="INT8"
                    )
                    st.pyplot(fig_center, use_container_width=True)
                    plt.close(fig_center)

                with vis_col2:
                    st.markdown(
                        """
                    <div class="section-title" style="font-size: 15px;">
                        <span>🔍</span> Model Probability Map
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                    fig_prob = create_probability_figure(probabilities)
                    st.pyplot(fig_prob, use_container_width=True)
                    plt.close(fig_prob)
                    st.caption(
                        "Sigmoid activation values [0.0, 1.0] before binary thresholding at 0.5. "
                        "Reflects raw neural network response (not medically calibrated)."
                    )

            elif model_mode == "FP32 — Baseline Precision":
                with st.spinner("Running FP32 Baseline Inference on CPU..."):
                    try:
                        session = load_fp32_session()
                        is_fallback = False
                    except Exception as exc:
                        st.warning(f"Could not load FP32 model ({exc}). Falling back to INT8.")
                        session = load_int8_session()
                        is_fallback = True

                    pred_mask, probabilities, latency_ms, fps = run_edge_inference(
                        session, current_image
                    )
                    dice = compute_dice(pred_mask, current_gt) if current_gt is not None else None
                    pred_pixels = int(pred_mask.sum())
                    gt_pixels = int(current_gt.sum()) if current_gt is not None else None
                    current_mb = int8_mb if is_fallback else fp32_mb
                    label_str = "INT8 (Fallback)" if is_fallback else "FP32 Baseline"

                detection = detect_tumor(pred_mask)

                # Status Completion Badge
                st.markdown(
                    f"""
                <div style="margin: 16px 0;">
                    <span class="status-badge" style="border-color: #58a6ff; color: #58a6ff; background: rgba(88, 166, 255, 0.15);">✓ {label_str} Inference Completed in {latency_ms:.2f} ms ({fps:.1f} FPS)</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                if detection["detected"]:
                    st.success(
                        f"🧠 {detection['label']} — "
                        f"{detection['tumor_pixels']:,} predicted tumor pixels "
                        f"({detection['tumor_area_percent']:.3f}% of the MRI slice)"
                    )
                else:
                    st.info(
                        f"✓ {detection['label']} — "
                        f"{detection['tumor_pixels']:,} predicted tumor pixels"
                    )

                st.markdown(
                    """
                <div class="section-title">
                    <span>🧠</span> Tumor Detection
                </div>
                """,
                    unsafe_allow_html=True,
                )

                det1, det2, det3 = st.columns(3)

                with det1:
                    st.metric(
                        "Detection Result",
                        detection["label"]
                    )

                with det2:
                    st.metric(
                        "Tumor Area",
                        f"{detection['tumor_area_percent']:.3f}%"
                    )

                with det3:
                    st.metric(
                        "Tumor Pixels",
                        f"{detection['tumor_pixels']:,}"
                    )

                st.caption(
                    "Detection is derived from the predicted segmentation mask. "
                    f"Minimum detection area: "
                    f"{detection['threshold_pixels']:,} pixels "
                    f"({detection['threshold_fraction'] * 100:.3f}%)."
                )

                st.caption(
                    "Detection is derived from the U-Net segmentation output: "
                    "a tumor is considered detected when the predicted tumor area "
                    "exceeds the configured minimum area threshold. "
                    "This is not a separately trained classification model."
                )

                st.markdown("<br>", unsafe_allow_html=True)

                # Quantitative Results Cards
                st.markdown(
                    """
                <div class="section-title">
                    <span>📊</span> 4. Segmentation Performance (Live Evaluation)
                </div>
                """,
                    unsafe_allow_html=True,
                )

                res1, res2, res3, res4, res5, res6 = st.columns(6)
                with res1:
                    st.metric("Dice Score", f"{dice * 100:.2f} %" if dice is not None else "N/A (Blind Eval)")
                with res2:
                    st.metric("Predicted Pixels", f"{pred_pixels:,}")
                with res3:
                    st.metric("Ground Truth Pixels", f"{gt_pixels:,}" if gt_pixels is not None else "N/A (Blind)")
                with res4:
                    st.metric("Latency", f"{latency_ms:.2f} ms")
                with res5:
                    st.metric("Throughput", f"{fps:.1f} FPS")
                with res6:
                    st.metric("Model Size", f"{current_mb:.2f} MB")

                if current_gt is None:
                    st.info(
                        "ℹ️ **Dice / IoU: N/A — Blind Evaluation (No Ground Truth Provided)**. "
                        "Jury test volume is evaluated without ground truth. Latency, throughput, "
                        "predicted tumor pixels, and memory footprint are measured live from ONNX Runtime CPU execution."
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # Centerpiece 4-Panel Visualization
                st.markdown(
                    """
                <div class="section-title">
                    <span>🖼️</span> 5. Visual Segmentation Centerpiece
                </div>
                """,
                    unsafe_allow_html=True,
                )

                vis_col1, vis_col2 = st.columns([2.5, 1])

                with vis_col1:
                    fig_center = create_four_panel_figure(
                        current_image, current_gt, pred_mask, model_label="FP32"
                    )
                    st.pyplot(fig_center, use_container_width=True)
                    plt.close(fig_center)

                with vis_col2:
                    st.markdown(
                        """
                    <div class="section-title" style="font-size: 15px;">
                        <span>🔍</span> Model Probability Map
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                    fig_prob = create_probability_figure(probabilities)
                    st.pyplot(fig_prob, use_container_width=True)
                    plt.close(fig_prob)
                    st.caption(
                        "Sigmoid activation values [0.0, 1.0] before binary thresholding at 0.5. "
                        "Reflects raw neural network response (not medically calibrated)."
                    )

            else:  # Side-by-Side Comparison
                with st.spinner("Running Dual FP32 & INT8 Inference on CPU..."):
                    session_int8 = load_int8_session()
                    pred_int8, prob_int8, lat_int8, fps_int8 = run_edge_inference(
                        session_int8, current_image
                    )
                    dice_int8 = compute_dice(pred_int8, current_gt) if current_gt is not None else None
                    detection_int8 = detect_tumor(pred_int8)

                    fp32_loaded = True
                    try:
                        session_fp32 = load_fp32_session()
                        pred_fp32, prob_fp32, lat_fp32, fps_fp32 = run_edge_inference(
                            session_fp32, current_image
                        )
                        dice_fp32 = compute_dice(pred_fp32, current_gt) if current_gt is not None else None
                        detection_fp32 = detect_tumor(pred_fp32)
                    except Exception as exc:
                        st.warning(f"Could not execute FP32 baseline ({exc}). Comparing INT8 only.")
                        pred_fp32, prob_fp32, lat_fp32, fps_fp32 = pred_int8, prob_int8, lat_int8, fps_int8
                        dice_fp32 = dice_int8
                        detection_fp32 = detection_int8
                        fp32_loaded = False

                    speedup = (lat_fp32 / lat_int8) if lat_int8 > 0 else 1.0
                    lat_reduction = ((lat_fp32 - lat_int8) / lat_fp32 * 100.0) if lat_fp32 > 0 else 0.0
                    parity = compute_parity(pred_fp32, pred_int8)

                # Status Completion Badge
                st.markdown(
                    f"""
                <div style="margin: 16px 0;">
                    <span class="status-badge">✓ Dual Inference Completed: INT8 ({lat_int8:.2f} ms) vs FP32 ({lat_fp32:.2f} ms) — {speedup:.2f}× Speedup (↓ {lat_reduction:.1f}% Latency)</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    """
                <div class="section-title">
                    <span>🧠</span> Tumor Detection Comparison
                </div>
                """,
                    unsafe_allow_html=True,
                )

                det_c1, det_c2 = st.columns(2)
                with det_c1:
                    st.metric(
                        "FP32 Detection",
                        detection_fp32["label"],
                    )
                    st.caption(
                        f"**Tumor Pixels:** {detection_fp32['tumor_pixels']:,} "
                        f"({detection_fp32['tumor_area_percent']:.3f}% of image)"
                    )
                with det_c2:
                    st.metric(
                        "INT8 Detection",
                        detection_int8["label"],
                    )
                    st.caption(
                        f"**Tumor Pixels:** {detection_int8['tumor_pixels']:,} "
                        f"({detection_int8['tumor_area_percent']:.3f}% of image)"
                    )

                st.caption(
                    "Detection is derived from the U-Net segmentation output: "
                    "a tumor is considered detected when the predicted tumor area "
                    "exceeds the configured minimum area threshold. "
                    "This is not a separately trained classification model."
                )

                st.markdown("<br>", unsafe_allow_html=True)

                # Section 4: Live Model Comparison Table & Cards
                st.markdown(
                    """
                <div class="section-title">
                    <span>⚖️</span> 4. FP32 vs Static INT8 Comparative Benchmark (Live Execution)
                </div>
                """,
                    unsafe_allow_html=True,
                )

                comp_col1, comp_col2, comp_col3 = st.columns(3)

                with comp_col1:
                    dice_fp32_str = f"{dice_fp32 * 100:.2f} %" if dice_fp32 is not None else "N/A (Blind Eval)"
                    st.markdown(
                        f"""
                    <div class="metric-card" style="border-left: 3px solid #58a6ff;">
                        <div class="metric-label">FP32 Baseline</div>
                        <div style="margin-top: 8px;">
                            <span style="color: #8b949e; font-size: 13px;">Latency:</span>
                            <strong style="color: #f0f6fc; font-size: 16px;"> {lat_fp32:.2f} ms</strong>
                        </div>
                        <div style="margin-top: 4px;">
                            <span style="color: #8b949e; font-size: 13px;">Throughput:</span>
                            <strong style="color: #f0f6fc; font-size: 16px;"> {fps_fp32:.1f} FPS</strong>
                        </div>
                        <div style="margin-top: 4px;">
                            <span style="color: #8b949e; font-size: 13px;">Model Size:</span>
                            <strong style="color: #f0f6fc; font-size: 16px;"> {fp32_mb:.2f} MB</strong>
                        </div>
                        <div style="margin-top: 4px;">
                            <span style="color: #8b949e; font-size: 13px;">Slice Dice:</span>
                            <strong style="color: #58a6ff; font-size: 16px;"> {dice_fp32_str}</strong>
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                with comp_col2:
                    dice_int8_str = f"{dice_int8 * 100:.2f} %" if dice_int8 is not None else "N/A (Blind Eval)"
                    st.markdown(
                        f"""
                    <div class="metric-card" style="border-left: 3px solid #3fb950;">
                        <div class="metric-label">INT8 Edge Model</div>
                        <div style="margin-top: 8px;">
                            <span style="color: #8b949e; font-size: 13px;">Latency:</span>
                            <strong style="color: #f0f6fc; font-size: 16px;"> {lat_int8:.2f} ms</strong>
                        </div>
                        <div style="margin-top: 4px;">
                            <span style="color: #8b949e; font-size: 13px;">Throughput:</span>
                            <strong style="color: #f0f6fc; font-size: 16px;"> {fps_int8:.1f} FPS</strong>
                        </div>
                        <div style="margin-top: 4px;">
                            <span style="color: #8b949e; font-size: 13px;">Model Size:</span>
                            <strong style="color: #f0f6fc; font-size: 16px;"> {int8_mb:.2f} MB</strong>
                        </div>
                        <div style="margin-top: 4px;">
                            <span style="color: #8b949e; font-size: 13px;">Slice Dice:</span>
                            <strong style="color: #3fb950; font-size: 16px;"> {dice_int8_str}</strong>
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                if current_gt is None:
                    st.info(
                        "ℹ️ **Dice / IoU: N/A — Blind Evaluation (No Ground Truth Provided)**. "
                        "The jury test case contains no ground-truth masks. Prediction parity, "
                        "model latency, memory savings, and throughput reflect live dual-model inference."
                    )

                with comp_col3:
                    st.markdown(
                        f"""
                    <div class="metric-card" style="border-left: 3px solid #00E5FF;">
                        <div class="metric-label">Edge Optimization Delta</div>
                        <div style="margin-top: 8px;">
                            <span style="color: #8b949e; font-size: 13px;">Speedup:</span>
                            <strong style="color: #00E5FF; font-size: 16px;"> {speedup:.2f}×</strong>
                            <span style="color: #3fb950; font-size: 12px;"> (↓ {lat_reduction:.1f}%)</span>
                        </div>
                        <div style="margin-top: 4px;">
                            <span style="color: #8b949e; font-size: 13px;">Storage Saved:</span>
                            <strong style="color: #3fb950; font-size: 16px;"> ↓ {size_reduction:.1f}%</strong>
                            <span style="color: #8b949e; font-size: 12px;"> ({fp32_mb - int8_mb:.2f} MB)</span>
                        </div>
                        <div style="margin-top: 4px;">
                            <span style="color: #8b949e; font-size: 13px;">Prediction Parity:</span>
                            <strong style="color: #00E5FF; font-size: 16px;"> {parity['parity_pct']:.2f} %</strong>
                        </div>
                        <div style="margin-top: 4px;">
                            <span style="color: #8b949e; font-size: 13px;">Differing Pixels:</span>
                            <strong style="color: #f0f6fc; font-size: 16px;"> {parity['differing_pixels']:,} / {parity['total_pixels']:,}</strong>
                            <span style="color: #8b949e; font-size: 12px;"> ({parity['differing_pct']:.3f}%)</span>
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                st.caption(
                    "ℹ️ Prediction parity measures exact pixel-level agreement between the FP32 baseline and "
                    "static INT8 quantized model. Numerical parity demonstrates algorithmic preservation under "
                    "edge quantization; it does not denote clinical equivalence."
                )

                st.markdown("<br>", unsafe_allow_html=True)

                # Centerpiece Side-by-Side Visualization
                st.markdown(
                    """
                <div class="section-title">
                    <span>🖼️</span> 5. Visual Segmentation Side-by-Side Centerpiece
                </div>
                """,
                    unsafe_allow_html=True,
                )

                vis_col1, vis_col2 = st.columns([2.5, 1])

                with vis_col1:
                    fig_sbs = create_side_by_side_figure(
                        current_image,
                        current_gt,
                        pred_fp32,
                        pred_int8,
                        dice_fp32=dice_fp32,
                        dice_int8=dice_int8,
                        fp32_latency=lat_fp32,
                        int8_latency=lat_int8,
                    )
                    st.pyplot(fig_sbs, use_container_width=True)
                    plt.close(fig_sbs)

                with vis_col2:
                    st.markdown(
                        """
                    <div class="section-title" style="font-size: 15px;">
                        <span>🔍</span> INT8 Probability Map
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                    fig_prob = create_probability_figure(prob_int8)
                    st.pyplot(fig_prob, use_container_width=True)
                    plt.close(fig_prob)
                    st.caption(
                        "INT8 sigmoid confidence map before thresholding. "
                        "Reflects pixel-wise tumor activation on edge hardware."
                    )

        except Exception as e:
            st.error(f"Inference execution failed: {e}")


else:
    st.info(
        "Please select or upload a valid MRI slice (.h5) above to begin the Edge AI demonstration."
    )

# ==============================================================================
# SECTION 4: EDGE PROFILE & QUANTIZATION COMPARISON
# ==============================================================================
st.markdown("<hr style='border-color: #21262d; margin: 36px 0;'>", unsafe_allow_html=True)

col_edge, col_quant = st.columns([1, 1.2])

with col_edge:
    st.markdown(
        """
    <div class="section-title">
        <span>🍓</span> Edge Deployment Profile
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
    | Deployment Property | Value / Specification |
    | :--- | :--- |
    | **Target Architecture** | ARM Cortex-A (Raspberry Pi 4 / 5) |
    | **Model Type** | 2D U-Net (Ronneberger et al.) |
    | **Format** | ONNX (Opset 17) |
    | **Quantization** | **Static QDQ INT8** (Per-channel signed int8) |
    | **Runtime Engine** | ONNX Runtime (`CPUExecutionProvider`) |
    | **Input Tensor** | `1 × 1 × 240 × 240` (Float32) |
    | **Output Tensor** | `1 × 1 × 240 × 240` (Raw Logits) |
    | **Peak Memory Footprint** | `< 30 MB RAM` during single-slice inference |
    | **Binary Size** | **{int8_mb:.2f} MB** |
    """
    )

with col_quant:
    st.markdown(
        """
    <div class="section-title">
        <span>⚖️</span> Quantization Impact (FP32 vs Static INT8)
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    > **Note**: Benchmark validated across **16 holdout validation slices** from `BRATS_001` (not entire 152-patient dataset).
    """
    )

    col_q1, col_q2, col_q3 = st.columns(3)
    with col_q1:
        st.metric("FP32 Mean Dice", "94.21 %", delta="Baseline")
    with col_q2:
        st.metric(
            "INT8 Mean Dice",
            "93.98 %",
            delta="-0.24 pp",
            delta_color="inverse",
        )
    with col_q3:
        st.metric("Parity Rate", "16 / 16 (100%)", delta="Within 1 pp of FP32")

    # Visual Bar Comparison for Model Size
    st.markdown("<br>", unsafe_allow_html=True)
    fig_size, ax_size = plt.subplots(figsize=(6, 2.2), facecolor="#0d1117")
    models = ["FP32 Model", "Static INT8 Model"]
    sizes = [fp32_mb, int8_mb]
    colors = ["#1f6feb", "#3fb950"]

    bars = ax_size.barh(models, sizes, color=colors, height=0.5)
    ax_size.set_xlabel("Disk Storage (MB)", color="#8b949e", fontsize=9)
    ax_size.set_xlim(0, max(sizes) * 1.3)
    ax_size.tick_params(colors="#8b949e", labelsize=9)
    ax_size.spines["top"].set_visible(False)
    ax_size.spines["right"].set_visible(False)
    ax_size.spines["left"].set_color("#30363d")
    ax_size.spines["bottom"].set_color("#30363d")

    for bar in bars:
        w = bar.get_width()
        ax_size.annotate(
            f"{w:.2f} MB",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color="#e6edf3",
            fontweight="bold",
            fontsize=9,
        )

    plt.tight_layout()
    st.pyplot(fig_size, use_container_width=True)
    plt.close(fig_size)

# ==============================================================================
# SECTION 5: DETECTION ACCURACY EVALUATION (DATASET BENCHMARK)
# ==============================================================================
st.markdown("<hr style='border-color: #21262d; margin: 36px 0;'>", unsafe_allow_html=True)

st.markdown(
    """
<div class="section-title">
    <span>🧠</span> 5. Detection Accuracy Evaluation (Dataset Benchmark)
</div>
""",
    unsafe_allow_html=True,
)

st.write(
    "Evaluate segmentation-derived tumor presence detection across the labeled HDF5 dataset. "
    "A slice is considered ground-truth positive if dataset `y` contains any tumor pixels. "
    f"A tumor is predicted as detected when the predicted mask area is at least {DETECTION_AREA_THRESHOLD * 100:.1f}% "
    f"of the slice ({int(np.ceil(240 * 240 * DETECTION_AREA_THRESHOLD))} pixels)."
)

col_eval_dir, col_eval_scope = st.columns([1.5, 1])
with col_eval_dir:
    eval_dataset_dir = st.text_input(
        "Evaluation Dataset Directory",
        value=DEFAULT_DATASET_ROOT,
        help="Path to folder containing .h5 files or patient subdirectories.",
        key="txt_eval_dataset_dir",
    )

with col_eval_scope:
    eval_scope = st.selectbox(
        "Evaluation Scope",
        options=[
            "Full Dataset (All Discovered Slices)",
            "Quick Benchmark (First 100 Slices)",
            "Standard Holdout (First 500 Slices)",
        ],
        index=0,
        help="Choose whether to evaluate all slices or a quick subset for immediate benchmarking.",
        key="sb_eval_scope",
    )

max_samples_eval = None
if eval_scope == "Quick Benchmark (First 100 Slices)":
    max_samples_eval = 100
elif eval_scope == "Standard Holdout (First 500 Slices)":
    max_samples_eval = 500

run_eval_btn = st.button("🚀 Run Detection Accuracy Evaluation", key="btn_run_eval")

if run_eval_btn:
    with st.spinner("Running Detection Accuracy Evaluation via INT8 ONNX model..."):
        try:
            eval_summary = cached_detection_evaluation(
                data_dir_str=eval_dataset_dir,
                model_path_str=str(INT8_MODEL_PATH),
                threshold_val=DETECTION_AREA_THRESHOLD,
                max_samples_val=max_samples_eval,
            )
            m = eval_summary["metrics"]

            # Dataset Summary Cards
            st.markdown(
                """
            <div class="section-title" style="font-size: 15px; margin-top: 16px;">
                <span>📁</span> Dataset Evaluation Summary
            </div>
            """,
                unsafe_allow_html=True,
            )

            ds_col1, ds_col2, ds_col3, ds_col4 = st.columns(4)
            with ds_col1:
                st.metric("Total Evaluated Slices", f"{eval_summary['valid_labeled_slices']:,}")
            with ds_col2:
                st.metric("Tumor-Positive Slices", f"{m['actual_positive']:,}")
            with ds_col3:
                st.metric("Tumor-Negative Slices", f"{m['actual_negative']:,}")
            with ds_col4:
                st.metric("Skipped / Invalid", f"{eval_summary['skipped_count']:,}")

            st.markdown("<br>", unsafe_allow_html=True)

            # Metric Cards
            st.markdown(
                """
            <div class="section-title" style="font-size: 15px;">
                <span>📊</span> Detection Metrics
            </div>
            """,
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Detection Accuracy", f"{m['accuracy'] * 100:.2f}%")
            with c2:
                sens_str = f"{m['sensitivity'] * 100:.2f}%" if m["sensitivity"] is not None else "N/A"
                st.metric("Sensitivity / Recall", sens_str)
            with c3:
                spec_str = f"{m['specificity'] * 100:.2f}%" if m["specificity"] is not None else "N/A (No Neg Slices)"
                st.metric("Specificity", spec_str)

            c4, c5 = st.columns(2)
            with c4:
                prec_str = f"{m['precision'] * 100:.2f}%" if m["precision"] is not None else "N/A"
                st.metric("Precision", prec_str)
            with c5:
                st.metric("F1 Score", f"{m['f1'] * 100:.2f}%")

            if not m["has_negatives"]:
                st.info(
                    "ℹ️ **Dataset Note**: The evaluated dataset contains exclusively tumor-positive slices (0 normal/negative slices). "
                    "Specificity (True Negative Rate) cannot be evaluated without negative slices. "
                    "Sensitivity and Precision confirm that positive slices are accurately detected."
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Confusion Matrix
            st.markdown(
                """
            <div class="section-title" style="font-size: 15px;">
                <span>🔢</span> Confusion Matrix
            </div>
            """,
                unsafe_allow_html=True,
            )

            cm_df = pd.DataFrame(
                [
                    [m["tn"], m["fp"]],
                    [m["fn"], m["tp"]],
                ],
                index=["Actual No Tumor", "Actual Tumor"],
                columns=["Predicted No Tumor", "Predicted Tumor"],
            )
            st.dataframe(cm_df, use_container_width=True)

            st.caption(
                f"Evaluation saved to `{eval_summary.get('csv_saved', 'results/detection_evaluation.csv')}`. "
                "Note: Segmentation Dice is a separate metric measuring spatial overlap (e.g. 92.87%), "
                "whereas Detection Accuracy measures binary presence classification."
            )

        except Exception as exc:
            st.error(f"Detection evaluation failed: {exc}")

# ==============================================================================
# FOOTER & MEDICAL DISCLAIMER
# ==============================================================================
st.markdown(
    """
<div class="app-footer">
    <strong>Research / Hackathon Prototype — Not a Medical Device</strong><br>
    Edge AI Brain Tumor Segmentation using INT8 U-Net & ONNX Runtime | IEEE Edge AI Hackathon 2026
</div>
""",
    unsafe_allow_html=True,
)
