import argparse
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

# Ensure repository root is on sys.path for modular imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure stdout handles Unicode characters on all platforms (including Windows consoles)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from monitoring.resource_monitor import (
    get_resource_snapshot,
    print_resource_report,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

REAL_TIME_THRESHOLD_MS = 200.0


# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------

def load_mri_and_mask(h5_path):
    """Load MRI slice and ground-truth mask safely."""

    if not os.path.isfile(h5_path):
        raise FileNotFoundError(
            f"MRI file not found: {h5_path}"
        )

    try:
        with h5py.File(h5_path, "r") as f:

            required_keys = {"x", "y"}

            missing_keys = (
                required_keys - set(f.keys())
            )

            if missing_keys:
                raise KeyError(
                    "Missing dataset(s): "
                    + ", ".join(sorted(missing_keys))
                )

            image = f["x"][:]
            mask = f["y"][:]

    except OSError as exc:
        raise OSError(
            f"Unable to read MRI HDF5 file: {h5_path}"
        ) from exc

    return image, mask


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

def preprocess(image):
    """Convert MRI to ONNX input format."""

    return image.astype(np.float32)[None, None, :, :]



def validate_mri(image, mask):
    """Validate MRI and ground-truth data before inference."""

    if image.ndim != 2:
        raise ValueError(
            f"Invalid MRI dimensions: {image.shape}. "
            "Expected a 2D image."
        )

    if image.shape != (240, 240):
        raise ValueError(
            f"Invalid MRI shape: {image.shape}. "
            "Expected (240, 240)."
        )

    if mask.shape != image.shape:
        raise ValueError(
            f"Image/mask shape mismatch: "
            f"{image.shape} vs {mask.shape}"
        )

    if not np.isfinite(image).all():
        raise ValueError(
            "MRI contains NaN or infinite values."
        )

    if not np.isfinite(mask).all():
        raise ValueError(
            "Ground-truth mask contains NaN or infinite values."
        )
# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------

def predict(session, image):
    """
    Run ONNX inference and measure inference latency.

    Returns:
        prediction: Binary tumor mask
        latency_ms: ONNX inference time in milliseconds
    """

    input_name = session.get_inputs()[0].name

    # Prepare input
    input_data = preprocess(image)

    # Measure ONLY model inference time.
    # Preprocessing is intentionally excluded.
    start_time = time.perf_counter()

    output = session.run(
        None,
        {input_name: input_data}
    )[0]

    end_time = time.perf_counter()

    latency_ms = (
        end_time - start_time
    ) * 1000.0

    # -----------------------------------------------------
    # Postprocessing
    # -----------------------------------------------------

    probability = 1.0 / (
        1.0 + np.exp(-output)
    )

    prediction = (
        probability > 0.5
    ).astype(np.uint8)[0, 0]

    return prediction, latency_ms


# ---------------------------------------------------------
# Dice
# ---------------------------------------------------------

def calculate_dice(prediction, ground_truth):
    """Calculate binary Dice score."""

    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    dice = (
        2.0 * intersection
        / (
            prediction.sum()
            + ground_truth.sum()
            + 1e-8
        )
    )

    return dice


# ---------------------------------------------------------
# Save Result
# ---------------------------------------------------------

def save_result(
    image,
    ground_truth,
    prediction,
    dice,
    model_type,
    output_path
):
    """Save MRI, ground truth and prediction comparison."""

    ground_truth_binary = (
        ground_truth > 0
    ).astype(np.uint8)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)

    plt.imshow(
        image,
        cmap="gray"
    )

    plt.title("MRI")
    plt.axis("off")

    plt.subplot(1, 3, 2)

    plt.imshow(
        ground_truth_binary,
        cmap="gray"
    )

    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)

    plt.imshow(
        prediction,
        cmap="gray"
    )

    plt.title(
        f"{model_type} Prediction - "
        f"Dice {dice * 100:.2f}%"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Brain Tumor Segmentation "
            "using ONNX U-Net"
        )
    )

    parser.add_argument(
        "--model",
        default="models/unet_int8.onnx"
    )

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--output",
        default="prediction.png"
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="ONNX Runtime CPU intra-op threads (default: 4)"
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Determine model type
    # -----------------------------------------------------

    model_type = (
        "INT8"
        if "int8" in args.model.lower()
        else "FP32"
    )

    print()
    print("=" * 55)
    print("      EDGE BRAIN TUMOR SEGMENTATION")
    print("=" * 55)

    print(f"Model       : {model_type}")
    print(f"Model path  : {args.model}")

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    print()
    print("Loading ONNX model...")

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = args.threads
    session_options.inter_op_num_threads = 1
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        args.model,
        sess_options=session_options,
        providers=["CPUExecutionProvider"]
    )

    print("Model loaded successfully.")

    # -----------------------------------------------------
    # Load MRI
    # -----------------------------------------------------

    image, ground_truth = load_mri_and_mask(
        args.input
    )
    validate_mri(
    image,
    ground_truth
)
    print()
    print("MRI shape   :", image.shape)
    print("MRI dtype   :", image.dtype)

    # -----------------------------------------------------
    # Run inference
    # -----------------------------------------------------

    prediction, inference_latency_ms = predict(
        session,
        image
    )

    # -----------------------------------------------------
    # Dice
    # -----------------------------------------------------

    ground_truth_binary = (
        ground_truth > 0
    ).astype(np.uint8)

    dice = calculate_dice(
        prediction,
        ground_truth_binary
    )

    # -----------------------------------------------------
    # Real-time evaluation
    # -----------------------------------------------------

    if inference_latency_ms <= REAL_TIME_THRESHOLD_MS:

        real_time_status = "✓ REAL-TIME TARGET MET"

    else:

        real_time_status = "⚠ ABOVE REAL-TIME TARGET"

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print()
    print("-" * 55)
    print("PERFORMANCE")
    print("-" * 55)

    print(
        f"Inference latency : "
        f"{inference_latency_ms:.2f} ms"
    )

    print(
        f"Latency threshold : "
        f"{REAL_TIME_THRESHOLD_MS:.2f} ms"
    )

    print(
        f"Status            : "
        f"{real_time_status}"
    )

    print()
    print("Predicted tumor pixels:",
          int(prediction.sum()))

    print(
        "Ground-truth tumor pixels:",
        int(ground_truth_binary.sum())
    )

    print(
        f"Dice score         : "
        f"{dice * 100:.2f}%"
    )

    # -----------------------------------------------------
    # System Resource Monitoring
    # -----------------------------------------------------

    input_data = preprocess(image)
    active_provider = session.get_providers()[0]

    resource_snapshot = get_resource_snapshot(
        model_path=args.model,
        input_array=input_data,
        output_array=prediction,
        execution_provider=active_provider,
        threads=args.threads,
    )

    print_resource_report(resource_snapshot)

    # -----------------------------------------------------
    # Save visualization
    # -----------------------------------------------------

    save_result(
        image,
        ground_truth,
        prediction,
        dice,
        model_type,
        args.output
    )

    print()
    print("Result saved:", args.output)
    print("=" * 55)


if __name__ == "__main__":

    try:
        main()

    except FileNotFoundError as exc:

        print()
        print("=" * 55)
        print("ERROR: FILE NOT FOUND")
        print("=" * 55)
        print(str(exc))
        print("Please check the supplied path.")
        print("=" * 55)

    except KeyError as exc:

        print()
        print("=" * 55)
        print("ERROR: INVALID HDF5 FILE")
        print("=" * 55)
        print(str(exc))
        print("Expected datasets: x and y")
        print("=" * 55)

    except ValueError as exc:

        print()
        print("=" * 55)
        print("ERROR: INVALID MRI DATA")
        print("=" * 55)
        print(str(exc))
        print("=" * 55)

    except OSError as exc:

        print()
        print("=" * 55)
        print("ERROR: FILE ACCESS FAILURE")
        print("=" * 55)
        print(str(exc))
        print("=" * 55)

    except Exception as exc:

        print()
        print("=" * 55)
        print("ERROR: INFERENCE FAILURE")
        print("=" * 55)
        print(str(exc))
        print("The system stopped safely.")
        print("=" * 55)