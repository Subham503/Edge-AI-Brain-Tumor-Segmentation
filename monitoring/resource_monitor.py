import os
import psutil


_CURRENT_PROCESS = None


def get_process():
    """Return the cached current Python process."""
    global _CURRENT_PROCESS
    if _CURRENT_PROCESS is None:
        _CURRENT_PROCESS = psutil.Process(os.getpid())
    return _CURRENT_PROCESS


def get_cpu_usage(interval=0.1):
    """Return current process CPU usage percentage."""
    process = get_process()

    # First call initializes the measurement
    process.cpu_percent(interval=None)

    return float(process.cpu_percent(interval=interval))


def get_process_memory_mb():
    """Return current process RAM usage in MB."""
    process = get_process()

    return float(process.memory_info().rss) / (1024 * 1024)


def get_system_memory():
    """Return system RAM statistics."""
    memory = psutil.virtual_memory()

    return {
        "total_mb": float(memory.total) / (1024 * 1024),
        "available_mb": float(memory.available) / (1024 * 1024),
        "used_mb": float(memory.used) / (1024 * 1024),
        "percent": float(memory.percent),
    }


def get_model_size_mb(model_path):
    """Return ONNX model size in MB."""
    if not model_path or not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    return float(os.path.getsize(model_path)) / (1024 * 1024)


def get_buffer_size_mb(array):
    """Return NumPy array memory size in MB."""
    if array is None or not hasattr(array, "nbytes"):
        return 0.0

    return float(array.nbytes) / (1024 * 1024)


def get_resource_snapshot(
    model_path,
    input_array=None,
    output_array=None,
    execution_provider=None,
    threads=None,
):
    """Collect a complete system resource snapshot."""
    system_memory = get_system_memory()

    return {
        "execution_provider": execution_provider,
        "threads": threads,
        "cpu_percent": get_cpu_usage(),
        "process_ram_mb": get_process_memory_mb(),
        "system_ram_total_mb": system_memory["total_mb"],
        "system_ram_available_mb": system_memory["available_mb"],
        "system_ram_used_mb": system_memory["used_mb"],
        "system_ram_percent": system_memory["percent"],
        "model_size_mb": get_model_size_mb(model_path),
        "input_buffer_mb": get_buffer_size_mb(input_array),
        "output_buffer_mb": get_buffer_size_mb(output_array),
    }


def print_resource_report(
    snapshot,
    execution_provider=None,
    threads=None,
):
    """Print resource information in a readable format."""
    ep = (
        execution_provider
        if execution_provider is not None
        else snapshot.get("execution_provider")
    )
    th = (
        threads
        if threads is not None
        else snapshot.get("threads")
    )

    print()
    print("=" * 55)
    print("              SYSTEM RESOURCE MONITOR")
    print("=" * 55)

    if ep:
        print(f"Execution provider : {ep}")

    if th is not None:
        print(f"Threads            : {th}")

    print()
    print("CPU")
    print("-" * 40)
    print(f"Process CPU        : {snapshot['cpu_percent']:.2f} %")

    print()
    print("MEMORY")
    print("-" * 40)
    print(f"Process RAM        : {snapshot['process_ram_mb']:.2f} MB")
    if "system_ram_total_mb" in snapshot:
        print(f"System RAM total   : {snapshot['system_ram_total_mb']:.2f} MB")
    print(f"System RAM used    : {snapshot['system_ram_used_mb']:.2f} MB")
    print(f"System RAM free    : {snapshot['system_ram_available_mb']:.2f} MB")
    print(f"System RAM usage   : {snapshot['system_ram_percent']:.2f} %")

    print()
    print("MODEL & BUFFERS")
    print("-" * 40)
    print(f"Model size         : {snapshot['model_size_mb']:.2f} MB")
    print(f"Input buffer       : {snapshot['input_buffer_mb']:.4f} MB")
    print(f"Output buffer      : {snapshot['output_buffer_mb']:.4f} MB")
    print("=" * 55)