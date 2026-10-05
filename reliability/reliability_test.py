import argparse
import os
import time

import h5py
import numpy as np
import onnxruntime as ort
import psutil


def load_mri(h5_path):
    """Load MRI image from HDF5 file."""

    if not os.path.isfile(h5_path):
        raise FileNotFoundError(
            f"MRI file not found: {h5_path}"
        )

    with h5py.File(h5_path, "r") as f:

        if "x" not in f:
            raise KeyError(
                "HDF5 file does not contain dataset 'x'."
            )

        image = f["x"][:]

    return image


def preprocess(image):
    """Convert MRI image to ONNX input format."""

    if image.shape != (240, 240):
        raise ValueError(
            f"Invalid image shape: {image.shape}. "
            "Expected (240, 240)."
        )

    if not np.isfinite(image).all():
        raise ValueError(
            "MRI contains NaN or infinite values."
        )

    return image.astype(
        np.float32
    )[None, None, :, :]


def run_inference(session, input_name, input_data):
    """Run one inference and return latency."""

    start = time.perf_counter()

    output = session.run(
        None,
        {input_name: input_data}
    )

    latency_ms = (
        time.perf_counter() - start
    ) * 1000.0

    # Make sure the model actually returned output.
    if not output:
        raise RuntimeError(
            "Model returned no output."
        )

    return latency_ms


def percentile(values, p):
    """Calculate percentile without requiring extra packages."""

    return float(
        np.percentile(
            np.array(values),
            p
        )
    )

def get_memory_mb():
    """Return current process RAM usage in MB."""

    process = psutil.Process(
        os.getpid()
    )

    return process.memory_info().rss / (
        1024 * 1024
    )

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Repeated inference reliability test "
            "for Edge AI brain tumor segmentation."
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
        "--runs",
        type=int,
        default=100
    )

    args = parser.parse_args()

    print()
    print("=" * 60)
    print("        EDGE AI RELIABILITY TEST")
    print("=" * 60)

    print(f"Model : {args.model}")
    print(f"Input : {args.input}")
    print(f"Runs  : {args.runs}")

    # -----------------------------------------------------
    # Load input
    # -----------------------------------------------------

    image = load_mri(args.input)

    input_data = preprocess(image)

    print()
    print("Input loaded successfully.")
    print("Input shape :", input_data.shape)
    print("Input dtype :", input_data.dtype)

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    print()
    print("Loading ONNX model...")

    session = ort.InferenceSession(
        args.model,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    print("Model loaded successfully.")
    print("Input name :", input_name)

    # -----------------------------------------------------
    # Warm-up
    # -----------------------------------------------------

    print()
    print("Warm-up inference...")

    session.run(
        None,
        {input_name: input_data}
    )

    print("Warm-up complete.")

    # -----------------------------------------------------
    # Reliability test
    # -----------------------------------------------------

    latencies = []
    failures = 0

    print()
    print("Starting repeated inference test...")
    print()

    memory_before_mb = get_memory_mb()
    peak_memory_mb = memory_before_mb

    for run_number in range(
        1,
        args.runs + 1
    ):

        try:

            latency_ms = run_inference(
                session,
                input_name,
                input_data
            )

            latencies.append(
                latency_ms
            )

            current_memory_mb = get_memory_mb()
            if current_memory_mb > peak_memory_mb:
                peak_memory_mb = current_memory_mb

        except Exception as exc:

            failures += 1

            print(
                f"Run {run_number}: FAILED - {exc}"
            )

        # Progress every 10 runs
        if (
            run_number % 10 == 0
            or run_number == args.runs
        ):

            print(
                f"Completed "
                f"{run_number}/{args.runs}"
            )

    memory_after_mb = get_memory_mb()
    if memory_after_mb > peak_memory_mb:
        peak_memory_mb = memory_after_mb

    memory_growth_mb = (
        memory_after_mb - memory_before_mb
    )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    successful_runs = len(latencies)

    print()
    print("=" * 60)
    print("              RELIABILITY RESULTS")
    print("=" * 60)

    print(
        f"Total runs       : {args.runs}"
    )

    print(
        f"Successful runs  : {successful_runs}"
    )

    print(
        f"Failed runs      : {failures}"
    )

    if latencies:

        average = float(
            np.mean(latencies)
        )

        median = float(
            np.median(latencies)
        )

        minimum = float(
            np.min(latencies)
        )

        maximum = float(
            np.max(latencies)
        )

        p95 = percentile(
            latencies,
            95
        )

        print()
        print("LATENCY")
        print("-" * 40)

        print(
            f"Average          : {average:.2f} ms"
        )

        print(
            f"Median           : {median:.2f} ms"
        )

        print(
            f"Minimum          : {minimum:.2f} ms"
        )

        print(
            f"Maximum          : {maximum:.2f} ms"
        )

        print(
            f"P95              : {p95:.2f} ms"
        )

    # -----------------------------------------------------
    # Memory usage
    # -----------------------------------------------------

    print()
    print("MEMORY USAGE")
    print("-" * 40)

    print(
        f"Initial RAM      : {memory_before_mb:.2f} MB"
    )

    print(
        f"Final RAM        : {memory_after_mb:.2f} MB"
    )

    print(
        f"Peak RAM         : {peak_memory_mb:.2f} MB"
    )

    print(
        f"RAM growth       : {memory_growth_mb:.2f} MB"
    )

    # -----------------------------------------------------
    # Reliability verdict
    # -----------------------------------------------------

    print()
    print("RELIABILITY")
    print("-" * 40)

    if failures == 0:

        print(
            "Status           : [PASS] ALL RUNS SUCCESSFUL"
        )

    else:

        print(
            "Status           : [FAIL] FAILURES DETECTED"
        )

    print("=" * 60)


if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print()
        print("=" * 60)
        print("ERROR: RELIABILITY TEST FAILED")
        print("=" * 60)
        print(str(exc))
        print("=" * 60)
