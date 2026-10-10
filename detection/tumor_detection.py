"""
Edge AI Brain Tumor Detection
-----------------------------
Standalone detection layer built on the existing INT8 ONNX U-Net
segmentation model.

Pipeline:
    H5 MRI slice -> INT8 U-Net -> binary segmentation mask
                 -> tumor area analysis -> detection decision

This does NOT retrain or modify the segmentation model.

Expected H5 datasets:
    x : MRI image
    y : ground-truth mask (optional for inference/detection)

Default model:
    models/unet_int8.onnx

Usage:
    python tumor_detection.py path/to/BRATS_241_68.h5

Optional:
    python tumor_detection.py path/to/sample.h5 --threshold 0.001
    python tumor_detection.py path/to/sample.h5 --output results/detection_result.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = Path("models/unet_int8.onnx")
DEFAULT_OUTPUT = Path("results/detection_result.png")
DEFAULT_H5 = Path(r"C:\Users\SUBHAM\Desktop\IEEE_Dataset\extracted\BRATS_241\BRATS_241_68.h5")

# Minimum predicted tumor area as a fraction of the MRI slice.
# 0.001 = 0.1% of a 240x240 image ~= 58 pixels.
DEFAULT_AREA_THRESHOLD = 0.001

# Neural-network segmentation threshold.
DEFAULT_PROBABILITY_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_h5_slice(h5_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load one MRI slice and optional ground-truth mask from an H5 file."""

    with h5py.File(h5_path, "r") as f:
        if "x" not in f:
            raise KeyError("H5 file does not contain required dataset 'x'.")

        image = np.asarray(f["x"])

        ground_truth = None
        if "y" in f:
            ground_truth = np.asarray(f["y"])

    image = np.squeeze(image).astype(np.float32)

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D MRI slice after squeezing, got shape {image.shape}."
        )

    if ground_truth is not None:
        ground_truth = np.squeeze(ground_truth)
        if ground_truth.shape != image.shape:
            raise ValueError(
                f"Image shape {image.shape} does not match "
                f"ground-truth shape {ground_truth.shape}."
            )
        ground_truth = (ground_truth > 0).astype(np.uint8)

    return image, ground_truth


def preprocess(image: np.ndarray) -> np.ndarray:
    """Match the exact preprocessing used by the verified inference pipeline."""
    return image.astype(np.float32)[None, None, :, :]


# ---------------------------------------------------------------------------
# INT8 inference
# ---------------------------------------------------------------------------

def create_session(model_path: Path) -> ort.InferenceSession:
    """Create an ONNX Runtime CPU session."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"INT8 model not found: {model_path}\n"
            f"Run this script from the project root or pass --model."
        )

    return ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )


def run_inference(
    session: ort.InferenceSession,
    image: np.ndarray,
    probability_threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Run INT8 U-Net inference and return mask, probabilities, latency, FPS."""

    input_name = session.get_inputs()[0].name
    input_tensor = preprocess(image)

    start = time.perf_counter()

    output = session.run(None, {input_name: input_tensor})[0]

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0

    # ONNX post-processing matching verified inference pipeline
    probability = 1.0 / (1.0 + np.exp(-output))

    prediction = (
        probability > probability_threshold
    ).astype(np.uint8)[0, 0]

    return prediction, probability[0, 0], elapsed_ms, fps


# ---------------------------------------------------------------------------
# Detection layer
# ---------------------------------------------------------------------------

def detect_tumor(
    prediction: np.ndarray,
    min_area_fraction: float = DEFAULT_AREA_THRESHOLD,
) -> dict:
    """
    Convert a segmentation mask into a tumor presence decision.

    Detection is derived from the predicted segmentation area.

    Returns:
        detected             : True / False
        label                : human-readable result
        tumor_pixels         : number of predicted tumor pixels
        total_pixels         : image size
        tumor_area_fraction  : tumor pixels / total pixels
        tumor_area_percent   : same value as percentage
        threshold_pixels     : minimum pixels required for detection
    """

    mask = np.asarray(prediction).astype(bool)

    if mask.ndim != 2:
        raise ValueError(
            f"Expected a 2D segmentation mask, got shape {mask.shape}."
        )

    if not 0.0 <= min_area_fraction <= 1.0:
        raise ValueError("Detection area threshold must be between 0 and 1.")

    tumor_pixels = int(prediction.sum())
    total_pixels = int(prediction.size)
    tumor_area_fraction = tumor_pixels / total_pixels

    threshold_pixels = max(
        1,
        int(np.ceil(total_pixels * min_area_fraction)),
    )

    detected = tumor_pixels >= threshold_pixels

    return {
        "detected": bool(detected),
        "label": "TUMOR DETECTED" if detected else "NO TUMOR DETECTED",
        "tumor_pixels": tumor_pixels,
        "total_pixels": total_pixels,
        "tumor_area_fraction": tumor_area_fraction,
        "tumor_area_percent": tumor_area_fraction * 100.0,
        "threshold_pixels": threshold_pixels,
        "threshold_fraction": min_area_fraction,
    }


# ---------------------------------------------------------------------------
# Optional evaluation
# ---------------------------------------------------------------------------

def compute_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> dict:
    """
    Compute segmentation Dice and tumor detection accuracy against ground truth.

    Returns:
        dice: Dice similarity coefficient (overlap F1 score)
        accuracy: True tumor tissue detection accuracy (Sensitivity/Recall: TP / GT)
        precision: Predicted tumor precision (TP / Pred)
    """
    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)

    intersection = int(np.logical_and(pred, gt).sum())
    gt_total = int(gt.sum())
    pred_total = int(pred.sum())
    denom = pred_total + gt_total

    dice = float((2.0 * intersection) / denom) if denom > 0 else 1.0
    accuracy = float(intersection / gt_total) if gt_total > 0 else 1.0
    precision = float(intersection / pred_total) if pred_total > 0 else 0.0

    return {
        "dice": dice,
        "accuracy": accuracy,
        "precision": precision,
    }


def compute_dice(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute Dice score when ground truth is available."""
    return compute_metrics(prediction, ground_truth)["dice"]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_visualization(
    image: np.ndarray,
    prediction: np.ndarray,
    probabilities: np.ndarray,
    detection: dict,
    output_path: Path,
    ground_truth: np.ndarray | None = None,
) -> None:
    """Save a 3/4-panel detection and segmentation visualization."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if ground_truth is not None:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.ravel()

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("MRI Input")

        axes[1].imshow(image, cmap="gray")
        axes[1].imshow(
            ground_truth,
            cmap="Reds",
            alpha=0.65,
            interpolation="nearest",
        )
        axes[1].set_title("Ground Truth")

        axes[2].imshow(image, cmap="gray")
        axes[2].imshow(
            prediction,
            cmap="Blues",
            alpha=0.65,
            interpolation="nearest",
        )
        axes[2].set_title(
            f"Predicted Segmentation\n{detection['tumor_pixels']:,} tumor pixels"
        )

        axes[3].imshow(probabilities, cmap="inferno", vmin=0, vmax=1)
        axes[3].set_title("Tumor Probability Map")

    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("MRI Input")

        axes[1].imshow(image, cmap="gray")
        axes[1].imshow(
            prediction,
            cmap="Blues",
            alpha=0.65,
            interpolation="nearest",
        )
        axes[1].set_title(
            f"Segmentation\n{detection['tumor_pixels']:,} tumor pixels"
        )

        axes[2].imshow(probabilities, cmap="inferno", vmin=0, vmax=1)
        axes[2].set_title("Tumor Probability Map")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{detection['label']} | "
        f"Tumor area: {detection['tumor_area_percent']:.3f}%",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect tumor presence from the existing INT8 U-Net segmentation output."
    )

    parser.add_argument(
        "h5_file",
        nargs="?",
        default=DEFAULT_H5,
        type=Path,
        help=f"Path to an H5 MRI slice containing dataset 'x' (default: {DEFAULT_H5}).",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Path to INT8 ONNX model (default: {DEFAULT_MODEL}).",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_AREA_THRESHOLD,
        help=(
            "Minimum tumor area fraction for detection "
            f"(default: {DEFAULT_AREA_THRESHOLD})."
        ),
    )

    parser.add_argument(
        "--probability-threshold",
        type=float,
        default=DEFAULT_PROBABILITY_THRESHOLD,
        help=(
            "Pixel probability threshold for segmentation "
            f"(default: {DEFAULT_PROBABILITY_THRESHOLD})."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Visualization output path (default: {DEFAULT_OUTPUT}).",
    )

    args = parser.parse_args()

    print("=" * 65)
    print("EDGE AI BRAIN TUMOR DETECTION")
    print("=" * 65)

    print(f"\nMRI input : {args.h5_file}")
    print(f"INT8 model: {args.model}")

    # Load data.
    image, ground_truth = load_h5_slice(args.h5_file)

    print(f"Image shape: {image.shape}")

    # Create ONNX session.
    session = create_session(args.model)

    # Run INT8 inference.
    prediction, probabilities, latency_ms, fps = run_inference(
        session,
        image,
        probability_threshold=args.probability_threshold,
    )

    # Detection layer.
    detection = detect_tumor(
        prediction,
        min_area_fraction=args.threshold,
    )

    # Optional evaluation metrics if ground truth exists.
    metrics = None
    if ground_truth is not None:
        metrics = compute_metrics(prediction, ground_truth)

    # Save visualization.
    save_visualization(
        image=image,
        prediction=prediction,
        probabilities=probabilities,
        detection=detection,
        output_path=args.output,
        ground_truth=ground_truth,
    )

    # Results.
    print("\n" + "-" * 65)
    print("TUMOR DETECTION RESULT")
    print("-" * 65)

    print(f"Detection result    : {detection['label']}")
    print(f"Tumor pixels        : {detection['tumor_pixels']:,}")
    print(f"Total image pixels  : {detection['total_pixels']:,}")
    print(
        f"Tumor area          : "
        f"{detection['tumor_area_percent']:.3f}%"
    )
    print(
        f"Detection threshold : "
        f"{detection['threshold_pixels']:,} pixels "
        f"({detection['threshold_fraction'] * 100:.3f}%)"
    )

    print("\n" + "-" * 65)
    print("EDGE INFERENCE")
    print("-" * 65)

    print(f"Runtime             : INT8 ONNX Runtime / CPU")
    print(f"Latency             : {latency_ms:.2f} ms")
    print(f"Throughput          : {fps:.2f} FPS")
    print(f"Probability cutoff  : {args.probability_threshold:.2f}")

    if metrics is not None:
        print(f"Detection Accuracy  : {metrics['accuracy'] * 100:.2f}%")
        print(f"Segmentation Dice   : {metrics['dice'] * 100:.2f}%")

    print(f"\nVisualization saved : {args.output}")

    print("\n" + "=" * 65)
    print("NOTE: Detection is derived from the segmentation mask.")
    print("This is a hackathon/research prototype, not a medical device.")
    print("=" * 65)


if __name__ == "__main__":
    main()
