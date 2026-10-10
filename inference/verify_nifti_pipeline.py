"""
Standalone Verification Script for NIfTI Reader and Edge AI ONNX Pipeline
==========================================================================
Executes Step 3 and Step 4 validation:
1. Validates NIfTI volume loading and header parsing.
2. Validates whole-volume normalization and slice extraction.
3. Tests both unet_fp32.onnx and unet_int8.onnx inference.
4. Processes all 155 slices, recording tumor area, latency, FPS, and FP32/INT8 parity.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import onnxruntime as ort

from inference.nifti_reader import (
    discover_evaluation_cases,
    get_slice,
    get_slice_2d,
    normalize_volume,
    read_nifti,
    read_nifti_volume,
)


def run_standalone_verification():
    print("=" * 75)
    print("STEP 3: STANDALONE NIFTI PIPELINE VERIFICATION")
    print("=" * 75)

    eval_data_dir = PROJECT_ROOT / "evaluation_data"
    cases = discover_evaluation_cases(eval_data_dir)
    print(f"\n[+] Discovered {len(cases)} jury evaluation cases:")
    for c in cases:
        rec_tag = " [RECOMMENDED - PRIMARY]" if c["is_recommended"] else " [INSPECTION ONLY]"
        print(f"  - Modality: {c['modality']:<6} | Case: {c['case_id']} | Size: {c['filesize_mb']} MB{rec_tag}")
        print(f"    Path: {c['path']}")

    # Select primary FLAIR case
    flair_case = next((c for c in cases if c["is_recommended"]), None)
    if flair_case is None:
        raise FileNotFoundError("No recommended FLAIR case found in evaluation_data!")

    flair_path = flair_case["path"]
    print(f"\n[+] Loading primary FLAIR volume: {flair_path.name}")
    
    # 1. Read volume and header
    vol_raw, meta = read_nifti(flair_path)
    print(f"  - Header validation:")
    print(f"    * Magic: {meta['magic']}")
    print(f"    * Datatype: {meta['datatype_name']} (code {meta['datatype_code']}, {meta['bitpix']} bitpix)")
    print(f"    * Voxel Spacing: {meta['voxel_spacing']}")
    print(f"    * Offset: {meta['vox_offset']} bytes")
    print(f"  - Raw Volume Shape: {vol_raw.shape}")
    print(f"  - Raw Volume Dtype: {vol_raw.dtype}")
    print(f"  - Raw Min / Max / Mean: {vol_raw.min()} / {vol_raw.max()} / {vol_raw.mean():.2f}")

    assert vol_raw.shape == (240, 240, 155), f"Unexpected shape {vol_raw.shape}"
    assert vol_raw.dtype == np.int16, f"Unexpected dtype {vol_raw.dtype}"

    # 2. Normalize volume
    print("\n[+] Normalizing volume with whole-volume z-score:")
    vol_norm = normalize_volume(vol_raw)
    print(f"  - Normalized Shape: {vol_norm.shape}")
    print(f"  - Normalized Dtype: {vol_norm.dtype}")
    print(f"  - Normalized Min / Max / Mean / Std: {vol_norm.min():.4f} / {vol_norm.max():.4f} / {vol_norm.mean():.4f} / {vol_norm.std():.4f}")

    assert vol_norm.shape == (240, 240, 155), "Normalized shape mismatch"
    assert vol_norm.dtype == np.float32, "Normalized dtype must be float32"

    # 3. Slice extraction
    sample_slice_idx = 70
    slice_tensor = get_slice(vol_norm, sample_slice_idx)
    print(f"\n[+] Extracting sample slice {sample_slice_idx}:")
    print(f"  - Tensor Shape: {slice_tensor.shape}")
    print(f"  - Tensor Dtype: {slice_tensor.dtype}")
    print(f"  - Tensor Min / Max: {slice_tensor.min():.4f} / {slice_tensor.max():.4f}")

    assert slice_tensor.shape == (1, 1, 240, 240), f"Expected [1, 1, 240, 240], got {slice_tensor.shape}"
    assert slice_tensor.dtype == np.float32, f"Expected float32, got {slice_tensor.dtype}"

    # 4. Load models and verify inference
    fp32_model_path = PROJECT_ROOT / "models" / "unet_fp32.onnx"
    int8_model_path = PROJECT_ROOT / "models" / "unet_int8.onnx"

    print(f"\n[+] Loading ONNX sessions:")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    session_fp32 = ort.InferenceSession(str(fp32_model_path), sess_options=opts)
    session_int8 = ort.InferenceSession(str(int8_model_path), sess_options=opts)

    inp_name_fp32 = session_fp32.get_inputs()[0].name
    inp_name_int8 = session_int8.get_inputs()[0].name
    print(f"  - FP32 input name: {inp_name_fp32}, shape: {session_fp32.get_inputs()[0].shape}")
    print(f"  - INT8 input name: {inp_name_int8}, shape: {session_int8.get_inputs()[0].shape}")

    # Single slice inference test
    t0 = time.perf_counter()
    out_fp32 = session_fp32.run(None, {inp_name_fp32: slice_tensor})[0]
    lat_fp32 = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    out_int8 = session_int8.run(None, {inp_name_int8: slice_tensor})[0]
    lat_int8 = (time.perf_counter() - t0) * 1000.0

    prob_fp32 = 1.0 / (1.0 + np.exp(-out_fp32[0, 0]))
    prob_int8 = 1.0 / (1.0 + np.exp(-out_int8[0, 0]))
    mask_fp32 = (prob_fp32 > 0.5).astype(np.uint8)
    mask_int8 = (prob_int8 > 0.5).astype(np.uint8)

    parity = ((mask_fp32 == mask_int8).sum() / mask_fp32.size) * 100.0

    print(f"\n[+] Sample Slice {sample_slice_idx} Single-Inference Result:")
    print(f"  - Output shape: {out_fp32.shape}")
    print(f"  - FP32 Tumor Pixels: {int(mask_fp32.sum()):4d} | Latency: {lat_fp32:.2f} ms")
    print(f"  - INT8 Tumor Pixels: {int(mask_int8.sum()):4d} | Latency: {lat_int8:.2f} ms")
    print(f"  - FP32 vs INT8 Parity: {parity:.2f}%")

    print("\n" + "=" * 75)
    print("STEP 4: FULL VOLUME (ALL 155 SLICES) INFERENCE TEST")
    print("=" * 75)

    all_results = []
    total_fp32_time = 0.0
    total_int8_time = 0.0

    print(f"Running inference across all 155 slices (0 to 154)...")
    for s_idx in range(155):
        s_tensor = get_slice(vol_norm, s_idx)

        # FP32 inference
        t0 = time.perf_counter()
        raw_fp32 = session_fp32.run(None, {inp_name_fp32: s_tensor})[0]
        dt_fp32 = (time.perf_counter() - t0) * 1000.0
        total_fp32_time += dt_fp32

        # INT8 inference
        t0 = time.perf_counter()
        raw_int8 = session_int8.run(None, {inp_name_int8: s_tensor})[0]
        dt_int8 = (time.perf_counter() - t0) * 1000.0
        total_int8_time += dt_int8

        # Post-processing
        p_f = 1.0 / (1.0 + np.exp(-raw_fp32[0, 0]))
        p_i = 1.0 / (1.0 + np.exp(-raw_int8[0, 0]))
        m_f = (p_f > 0.5).astype(np.uint8)
        m_i = (p_i > 0.5).astype(np.uint8)

        cnt_f = int(m_f.sum())
        cnt_i = int(m_i.sum())
        par = ((m_f == m_i).sum() / m_f.size) * 100.0

        all_results.append({
            "slice": s_idx,
            "tumor_pixels_fp32": cnt_f,
            "tumor_pixels_int8": cnt_i,
            "lat_fp32_ms": dt_fp32,
            "lat_int8_ms": dt_int8,
            "fps_int8": 1000.0 / dt_int8 if dt_int8 > 0 else 0.0,
            "parity_pct": par,
        })

    tumor_bearing_slices = [r for r in all_results if r["tumor_pixels_int8"] > 50]
    mean_lat_fp32 = total_fp32_time / 155.0
    mean_lat_int8 = total_int8_time / 155.0
    mean_parity = float(np.mean([r["parity_pct"] for r in all_results]))

    print(f"\n[+] Full Volume 155 Slices Summary:")
    print(f"  - Total Slices Processed: {len(all_results)} / 155")
    print(f"  - Slices with Significant Tumor Activity (>50 px): {len(tumor_bearing_slices)}")
    if tumor_bearing_slices:
        span_start = tumor_bearing_slices[0]["slice"]
        span_end = tumor_bearing_slices[-1]["slice"]
        print(f"  - Tumor Active Span: Slices {span_start} to {span_end}")
        max_slice = max(tumor_bearing_slices, key=lambda x: x["tumor_pixels_int8"])
        print(f"  - Peak Tumor Slice: Slice {max_slice['slice']} (INT8: {max_slice['tumor_pixels_int8']} px, FP32: {max_slice['tumor_pixels_fp32']} px)")
    print(f"  - Average FP32 Latency: {mean_lat_fp32:.2f} ms")
    print(f"  - Average INT8 Latency: {mean_lat_int8:.2f} ms")
    print(f"  - INT8 Speedup: {mean_lat_fp32 / mean_lat_int8:.2f}x")
    print(f"  - Overall Average Parity: {mean_parity:.3f}%")
    print("\n[SUCCESS] Standalone NIfTI pipeline verified cleanly!")


if __name__ == "__main__":
    run_standalone_verification()
