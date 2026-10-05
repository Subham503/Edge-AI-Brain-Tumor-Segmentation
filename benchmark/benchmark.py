
"""
Raspberry Pi Edge-AI Benchmark
------------------------------
Benchmarks FP32 vs INT8 ONNX models using ONNX Runtime.

Metrics:
- Average inference latency
- Median inference latency
- Minimum / maximum latency
- P95 latency
- Approximate FPS
- CPU usage
- RAM usage
- Model size

Usage:
    python benchmark.py \
        --fp32 ../models/unet_fp32.onnx \
        --int8 ../models/unet_int8.onnx \
        --input ../data/sample.npy \
        --runs 100

Expected input:
    NumPy .npy file containing a single MRI slice.

Supported shapes:
    (240, 240)
    (1, 240, 240)
    (1, 1, 240, 240)

The script automatically converts the input to:
    (1, 1, 240, 240)

NOTE:
This benchmark measures inference only.
Preprocessing/postprocessing are excluded from latency.
"""

import argparse
import csv
import os
import time
import h5py

import numpy as np
import onnxruntime as ort

try:
    import psutil
except ImportError:
    psutil = None


# ---------------------------------------------------------
# Input loading
# ---------------------------------------------------------

def load_input(input_path):
    """Load MRI input from a BraTS .h5 file."""

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    with h5py.File(input_path, "r") as f:
        if "x" not in f:
            raise KeyError(
                f"'x' dataset not found in {input_path}. "
                f"Available keys: {list(f.keys())}"
            )

        image = f["x"][:]

    print(f"\nOriginal input shape : {image.shape}")
    print(f"Original input dtype : {image.dtype}")

    image = image.astype(np.float32)

    # Convert to NCHW
    if image.ndim == 2:
        # (H, W)
        image = image[np.newaxis, np.newaxis, :, :]

    elif image.ndim == 3:
        # (H, W, C) or (C, H, W)
        #
        # Your BraTS slices are expected to already contain
        # the channel dimension needed by the model.
        #
        # If shape is (H, W, C), convert to (1, C, H, W).
        if image.shape[-1] <= 4:
            image = np.transpose(image, (2, 0, 1))
            image = image[np.newaxis, :, :, :]
        else:
            image = image[np.newaxis, :, :, :]

    elif image.ndim == 4:
        pass

    else:
        raise ValueError(
            f"Unsupported input shape: {image.shape}"
        )

    print(f"Benchmark input shape: {image.shape}")
    print(f"Benchmark input dtype: {image.dtype}")

    return image


# ---------------------------------------------------------
# Model information
# ---------------------------------------------------------

def model_size_mb(model_path):
    """Return model size in MB."""

    size_bytes = os.path.getsize(model_path)

    return size_bytes / (1024 * 1024)


def get_model_input_name(session):
    """Get ONNX model input name."""

    return session.get_inputs()[0].name


# ---------------------------------------------------------
# Benchmark
# ---------------------------------------------------------

def benchmark_model(
    model_path,
    input_data,
    runs=100,
    warmup=10,
    threads=1,
):
    """Benchmark a single ONNX model."""

    print("\n" + "=" * 60)
    print(f"MODEL: {os.path.basename(model_path)}")
    print("=" * 60)

    # ---------------------------------------------
    # ONNX Runtime configuration
    # ---------------------------------------------

    session_options = ort.SessionOptions()

    # Keep thread count controlled for edge benchmarking.
    session_options.intra_op_num_threads = threads
    session_options.inter_op_num_threads = 1

    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    input_name = get_model_input_name(session)

    print(f"Input name        : {input_name}")
    print(f"Execution provider: CPUExecutionProvider")
    print(f"Threads           : {threads}")
    print(f"Model size        : {model_size_mb(model_path):.2f} MB")

    # ---------------------------------------------
    # Warm-up
    # ---------------------------------------------

    print(f"\nWarm-up runs: {warmup}")

    for _ in range(warmup):
        session.run(None, {input_name: input_data})

    # ---------------------------------------------
    # Process information
    # ---------------------------------------------

    process = None

    if psutil is not None:
        process = psutil.Process(os.getpid())

        # Establish CPU measurement baseline
        process.cpu_percent(interval=None)

    # ---------------------------------------------
    # Actual benchmark
    # ---------------------------------------------

    latencies = []

    if process is not None:
        cpu_start = process.cpu_times()
        wall_start = time.perf_counter()

    for _ in range(runs):

        start = time.perf_counter()

        session.run(
            None,
            {input_name: input_data}
        )

        end = time.perf_counter()

        latency_ms = (end - start) * 1000

        latencies.append(latency_ms)

    if process is not None:
        wall_end = time.perf_counter()
        cpu_end = process.cpu_times()

    # ---------------------------------------------
    # Statistics
    # ---------------------------------------------

    latencies = np.array(latencies)

    average_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)

    p95_latency = np.percentile(latencies, 95)

    fps = 1000 / average_latency

    # ---------------------------------------------
    # CPU usage
    # ---------------------------------------------

    cpu_usage = None

    if process is not None:

        cpu_time = (
            (cpu_end.user - cpu_start.user)
            + (cpu_end.system - cpu_start.system)
        )

        wall_time = wall_end - wall_start

        if wall_time > 0:
            cpu_count = psutil.cpu_count(logical=True)

            cpu_usage = (
                cpu_time
                / wall_time
                / cpu_count
                * 100
            )

    # ---------------------------------------------
    # RAM
    # ---------------------------------------------

    ram_mb = None

    if process is not None:

        memory_info = process.memory_info()

        ram_mb = memory_info.rss / (1024 * 1024)

    # ---------------------------------------------
    # Print results
    # ---------------------------------------------

    print("\nRESULTS")
    print("-" * 40)

    print(f"Runs               : {runs}")
    print(f"Average latency    : {average_latency:.3f} ms")
    print(f"Median latency     : {median_latency:.3f} ms")
    print(f"Minimum latency    : {min_latency:.3f} ms")
    print(f"Maximum latency    : {max_latency:.3f} ms")
    print(f"P95 latency        : {p95_latency:.3f} ms")
    print(f"Approx. FPS        : {fps:.2f}")

    if cpu_usage is not None:
        print(f"CPU usage          : {cpu_usage:.2f}%")
    else:
        print("CPU usage          : psutil not installed")

    if ram_mb is not None:
        print(f"Process RAM        : {ram_mb:.2f} MB")
    else:
        print("Process RAM        : psutil not installed")

    print(f"Model size         : {model_size_mb(model_path):.2f} MB")

    return {
        "model": os.path.basename(model_path),
        "model_size_mb": round(model_size_mb(model_path), 3),
        "average_latency_ms": round(average_latency, 3),
        "median_latency_ms": round(median_latency, 3),
        "min_latency_ms": round(min_latency, 3),
        "max_latency_ms": round(max_latency, 3),
        "p95_latency_ms": round(p95_latency, 3),
        "fps": round(fps, 3),
        "cpu_percent": (
            round(cpu_usage, 3)
            if cpu_usage is not None
            else ""
        ),
        "ram_mb": (
            round(ram_mb, 3)
            if ram_mb is not None
            else ""
        ),
        "runs": runs,
        "threads": threads,
    }


# ---------------------------------------------------------
# Save CSV
# ---------------------------------------------------------

def save_results(results, output_path):

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True
    )

    fieldnames = [
        "model",
        "model_size_mb",
        "average_latency_ms",
        "median_latency_ms",
        "min_latency_ms",
        "max_latency_ms",
        "p95_latency_ms",
        "fps",
        "cpu_percent",
        "ram_mb",
        "runs",
        "threads",
    ]

    with open(
        output_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:
            writer.writerow(result)

    print(f"\nResults saved to: {output_path}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Benchmark FP32 vs INT8 ONNX models."
    )

    parser.add_argument(
        "--fp32",
        required=True,
        help="Path to FP32 ONNX model"
    )

    parser.add_argument(
        "--int8",
        required=True,
        help="Path to INT8 ONNX model"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input .npy file"
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of benchmark runs"
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warm-up runs"
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="ONNX Runtime CPU threads"
    )

    parser.add_argument(
        "--output",
        default="../results/benchmark_results.csv",
        help="CSV output path"
    )

    args = parser.parse_args()

    # ---------------------------------------------
    # Load input
    # ---------------------------------------------

    input_data = load_input(args.input)

    # ---------------------------------------------
    # Benchmark both models
    # ---------------------------------------------

    results = []

    fp32_result = benchmark_model(
        args.fp32,
        input_data,
        runs=args.runs,
        warmup=args.warmup,
        threads=args.threads,
    )

    results.append(fp32_result)

    int8_result = benchmark_model(
        args.int8,
        input_data,
        runs=args.runs,
        warmup=args.warmup,
        threads=args.threads,
    )

    results.append(int8_result)

    # ---------------------------------------------
    # Save
    # ---------------------------------------------

    save_results(
        results,
        args.output
    )

    # ---------------------------------------------
    # Comparison
    # ---------------------------------------------

    print("\n" + "=" * 60)
    print("FP32 vs INT8 COMPARISON")
    print("=" * 60)

    fp32_latency = fp32_result["average_latency_ms"]
    int8_latency = int8_result["average_latency_ms"]

    fp32_size = fp32_result["model_size_mb"]
    int8_size = int8_result["model_size_mb"]

    if fp32_latency > 0:

        latency_change = (
            (fp32_latency - int8_latency)
            / fp32_latency
            * 100
        )

    else:
        latency_change = 0

    if fp32_size > 0:

        size_reduction = (
            (fp32_size - int8_size)
            / fp32_size
            * 100
        )

    else:
        size_reduction = 0

    print(f"\nFP32 latency : {fp32_latency:.3f} ms")
    print(f"INT8 latency : {int8_latency:.3f} ms")

    print(
        f"Latency improvement: "
        f"{latency_change:.2f}%"
    )

    print(f"\nFP32 size : {fp32_size:.2f} MB")
    print(f"INT8 size : {int8_size:.2f} MB")

    print(
        f"Model size reduction: "
        f"{size_reduction:.2f}%"
    )

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()

