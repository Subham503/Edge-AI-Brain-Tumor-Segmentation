"""
Edge AI Tumor Detection Evaluation Module
-----------------------------------------
Evaluates tumor presence detection accuracy across a dataset of HDF5 MRI slices
using the verified static INT8 ONNX U-Net segmentation pipeline.

Detection is derived from the predicted segmentation mask:
    pred_area_fraction >= DETECTION_AREA_THRESHOLD (default: 0.001 / 0.1% area)

Ground-truth label is positive if ANY tumor pixels exist in dataset 'y'.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import onnxruntime as ort

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Default configuration
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "unet_int8.onnx"
DEFAULT_DATASET_ROOT = r"C:\Users\SUBHAM\Desktop\IEEE_Dataset\extracted"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "detection_evaluation.csv"
DETECTION_AREA_THRESHOLD = 0.001


def preprocess(image: np.ndarray) -> np.ndarray:
    """
    Exact verified preprocessing used by run_edge_inference() and inference.py.
    Converts 2D MRI slice to NCHW float32 tensor without min-max normalization.
    """
    return image.astype(np.float32)[None, None, :, :]


def create_onnx_session(model_path: Path | str, threads: int = 4) -> ort.InferenceSession:
    """Create an ONNX Runtime CPU session matching edge runtime parameters."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model file not found at: {model_path}")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    return session


def compute_metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
    """
    Compute binary classification metrics safely handling division by zero.
    """
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    actual_pos = tp + fn
    actual_neg = tn + fp
    pred_pos = tp + fp

    sensitivity = tp / actual_pos if actual_pos > 0 else None
    specificity = tn / actual_neg if actual_neg > 0 else None
    precision = tp / pred_pos if pred_pos > 0 else None

    if precision is not None and sensitivity is not None and (precision + sensitivity) > 0:
        f1 = 2.0 * precision * sensitivity / (precision + sensitivity)
    else:
        f1 = 0.0

    return {
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "actual_positive": actual_pos,
        "actual_negative": actual_neg,
        "predicted_positive": pred_pos,
        "predicted_negative": tn + fn,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "has_negatives": actual_neg > 0,
        "has_positives": actual_pos > 0,
    }


def evaluate_slice(
    session: ort.InferenceSession,
    h5_path: Path,
    threshold: float = DETECTION_AREA_THRESHOLD,
) -> dict:
    """
    Evaluate a single H5 slice:
    - Reads 'x' and 'y'
    - Runs INT8 inference
    - Derives ground-truth and predicted detection labels
    - Classifies as TP, TN, FP, or FN
    """
    with h5py.File(h5_path, "r") as f:
        if "x" not in f or "y" not in f:
            raise KeyError("H5 file must contain both 'x' and 'y' datasets.")

        x = np.asarray(f["x"])
        y = np.asarray(f["y"])

    x_2d = np.squeeze(x).astype(np.float32)
    y_2d = np.squeeze(y)

    if x_2d.ndim != 2:
        raise ValueError(f"Expected 2D MRI slice, got {x_2d.shape}")

    # Ground truth: Positive if contains ANY tumor pixels
    gt_mask = (y_2d > 0).astype(np.uint8)
    gt_tumor_pixels = int(gt_mask.sum())
    actual_positive = bool(gt_tumor_pixels > 0)

    # Input tensor
    input_name = session.get_inputs()[0].name
    input_tensor = preprocess(x_2d)

    # Run inference
    start_time = time.perf_counter()
    raw_output = session.run(None, {input_name: input_tensor})[0]
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # Verified post-processing
    probability = 1.0 / (1.0 + np.exp(-raw_output))
    pred_mask = (probability > 0.5).astype(np.uint8)[0, 0]

    # Segmentation-derived detection
    pred_tumor_pixels = int(pred_mask.sum())
    total_pixels = int(pred_mask.size)
    pred_area_fraction = pred_tumor_pixels / total_pixels if total_pixels > 0 else 0.0

    predicted_positive = bool(pred_area_fraction >= threshold)

    # Classification
    if actual_positive and predicted_positive:
        cat = "TP"
    elif not actual_positive and not predicted_positive:
        cat = "TN"
    elif not actual_positive and predicted_positive:
        cat = "FP"
    else:
        cat = "FN"

    # Dice score
    intersection = np.logical_and(pred_mask > 0, gt_mask > 0).sum()
    dice = float((2.0 * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-8))

    return {
        "file": h5_path.name,
        "path": str(h5_path),
        "actual": "TUMOR" if actual_positive else "NO TUMOR",
        "predicted": "TUMOR" if predicted_positive else "NO TUMOR",
        "gt_tumor_pixels": gt_tumor_pixels,
        "pred_tumor_pixels": pred_tumor_pixels,
        "pred_area_percent": pred_area_fraction * 100.0,
        "dice": dice,
        "latency_ms": latency_ms,
        "correct": bool(actual_positive == predicted_positive),
        "classification": cat,
    }


def evaluate_dataset(
    data_dir: Path | str,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    threshold: float = DETECTION_AREA_THRESHOLD,
    max_samples: int | None = None,
    output_csv: Path | str | None = DEFAULT_OUTPUT_CSV,
    threads: int = 4,
    progress_callback: callable | None = None,
) -> dict:
    """
    Run full dataset tumor detection evaluation.
    Scans data_dir recursively for .h5 files, evaluates INT8 detection,
    computes confusion matrix metrics, and optionally saves per-slice CSV.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")

    # Discover H5 files
    discovered_files = sorted(list(data_path.rglob("*.h5")))
    total_discovered = len(discovered_files)

    if max_samples is not None and max_samples > 0:
        files_to_eval = discovered_files[:max_samples]
    else:
        files_to_eval = discovered_files

    # Load ONNX session
    session = create_onnx_session(model_path, threads=threads)

    results = []
    skipped_files = []
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    total_files = len(files_to_eval)
    for idx, h5_file in enumerate(files_to_eval):
        try:
            res = evaluate_slice(session, h5_file, threshold=threshold)
            results.append(res)
            cat = res["classification"]
            if cat == "TP":
                tp += 1
            elif cat == "TN":
                tn += 1
            elif cat == "FP":
                fp += 1
            elif cat == "FN":
                fn += 1
        except Exception as exc:
            skipped_files.append({"file": h5_file.name, "path": str(h5_file), "error": str(exc)})

        if progress_callback is not None:
            progress_callback(idx + 1, total_files)

    metrics = compute_metrics(tp, tn, fp, fn)

    summary = {
        "dataset_path": str(data_path),
        "model_path": str(model_path),
        "threshold": threshold,
        "files_discovered": total_discovered,
        "files_evaluated": len(results),
        "valid_labeled_slices": len(results),
        "skipped_files": skipped_files,
        "skipped_count": len(skipped_files),
        "metrics": metrics,
        "results": results,
    }

    # Save CSV if requested
    if output_csv is not None and results:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "file",
                "actual",
                "predicted",
                "gt_tumor_pixels",
                "pred_tumor_pixels",
                "pred_area_percent",
                "correct",
                "classification",
                "dice",
                "latency_ms",
            ])
            for r in results:
                writer.writerow([
                    r["file"],
                    r["actual"],
                    r["predicted"],
                    r["gt_tumor_pixels"],
                    r["pred_tumor_pixels"],
                    f"{r['pred_area_percent']:.4f}",
                    r["correct"],
                    r["classification"],
                    f"{r['dice']:.4f}",
                    f"{r['latency_ms']:.2f}",
                ])
        summary["csv_saved"] = str(csv_path)

    return summary


def print_cli_report(summary: dict) -> None:
    """Format and print the evaluation report matching the project specification."""
    m = summary["metrics"]
    thresh = summary["threshold"]
    thresh_pct = thresh * 100.0

    print()
    print("=" * 60)
    print("EDGE AI TUMOR DETECTION EVALUATION")
    print("=" * 60)
    print()
    print("Model:")
    print(summary["model_path"])
    print()
    print("Dataset:")
    print(summary["dataset_path"])
    print()
    print("Detection threshold:")
    print(f"{thresh_pct:.3f}%")
    print()
    print("-" * 60)
    print("DATASET")
    print("-" * 60)
    print(f"Files discovered       : {summary['files_discovered']:,}")
    print(f"Valid labeled slices   : {summary['valid_labeled_slices']:,}")
    print(f"Skipped                : {summary['skipped_count']:,}")
    print()
    print(f"Actual tumor slices    : {m['actual_positive']:,}")
    print(f"Actual normal slices   : {m['actual_negative']:,}")
    print()
    print("-" * 60)
    print("CONFUSION MATRIX")
    print("-" * 60)
    print()
    print("                     Predicted")
    print("                  No Tumor   Tumor")
    print()
    print(f"Actual No Tumor   {m['tn']:>8}  {m['fp']:>6}")
    print()
    print(f"Actual Tumor      {m['fn']:>8}  {m['tp']:>6}")
    print()
    print("-" * 60)
    print("DETECTION METRICS")
    print("-" * 60)
    print(f"Accuracy             : {m['accuracy'] * 100:.2f}%")

    if m["sensitivity"] is not None:
        print(f"Sensitivity / Recall : {m['sensitivity'] * 100:.2f}%")
    else:
        print("Sensitivity / Recall : N/A (No positive slices)")

    if m["specificity"] is not None:
        print(f"Specificity          : {m['specificity'] * 100:.2f}%")
    else:
        print("Specificity          : N/A (No negative slices in dataset)")

    if m["precision"] is not None:
        print(f"Precision            : {m['precision'] * 100:.2f}%")
    else:
        print("Precision            : N/A (No positive predictions)")

    print(f"F1 Score             : {m['f1'] * 100:.2f}%")

    if not m["has_negatives"]:
        print()
        print("NOTE: The evaluated dataset contains ONLY tumor-positive slices.")
        print("Specificity (TN / (TN + FP)) cannot be meaningfully evaluated without")
        print("negative slices. Sensitivity and Precision evaluate positive detection.")

    if summary.get("csv_saved"):
        print()
        print("-" * 60)
        print("OUTPUT")
        print("-" * 60)
        print(f"CSV saved:")
        print(summary["csv_saved"])
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Rigorous Edge AI Brain Tumor Detection Accuracy Evaluation"
    )
    parser.add_argument(
        "dataset_pos",
        nargs="?",
        default=None,
        help=f"Dataset path (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Dataset path (alternative to positional argument)",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help=f"Path to INT8 ONNX model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DETECTION_AREA_THRESHOLD,
        help=f"Detection area threshold fraction (default: {DETECTION_AREA_THRESHOLD})",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_CSV),
        help=f"Path to output CSV (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on maximum slices to evaluate (for testing)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="ONNX intra-op CPU threads (default: 4)",
    )

    args = parser.parse_args()

    data_dir = args.data_dir or args.dataset_pos or DEFAULT_DATASET_ROOT

    t_start = time.perf_counter()
    summary = evaluate_dataset(
        data_dir=data_dir,
        model_path=args.model,
        threshold=args.threshold,
        max_samples=args.max_samples,
        output_csv=args.output_csv,
        threads=args.threads,
    )
    total_time = time.perf_counter() - t_start

    print_cli_report(summary)
    print(f"\nEvaluation completed in {total_time:.2f} seconds.")


if __name__ == "__main__":
    main()
