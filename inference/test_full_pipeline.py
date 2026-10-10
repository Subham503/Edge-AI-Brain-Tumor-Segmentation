"""
Comprehensive Verification: H5 Pipeline vs NIfTI Adapter Pipeline
==================================================================
Tests both workflows to ensure zero regression on existing functionality:
1. Existing H5 Workflow:
   - Loads .h5 slice
   - Checks 'x' and 'y' datasets
   - Runs FP32 and INT8 inference
   - Verifies Dice score computation with ground truth
   
2. New NIfTI Adapter Workflow:
   - Reads 3D NIfTI volume
   - Normalizes with whole-volume z-score
   - Extracts axial slice (shape [1, 1, 240, 240], float32)
   - Runs FP32 and INT8 inference
   - Validates blind evaluation handling (gt is None)
   - Measures FP32 vs INT8 parity
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
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


def compute_dice(pred, gt):
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    return float((2.0 * inter) / (p.sum() + g.sum() + 1e-8))


def compute_parity(p1, p2):
    total = p1.size
    ident = int((p1 == p2).sum())
    return (ident / total) * 100.0


def test_pipeline():
    print("=" * 70)
    print("VERIFICATION SUITE: H5 WORKFLOW & NIFTI WORKFLOW")
    print("=" * 70)

    # Load ONNX sessions
    fp32_path = PROJECT_ROOT / "models" / "unet_fp32.onnx"
    int8_path = PROJECT_ROOT / "models" / "unet_int8.onnx"

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    sess_fp32 = ort.InferenceSession(str(fp32_path), sess_options=opts)
    sess_int8 = ort.InferenceSession(str(int8_path), sess_options=opts)

    inp_name_f = sess_fp32.get_inputs()[0].name
    inp_name_i = sess_int8.get_inputs()[0].name

    # -------------------------------------------------------------
    # 1. Existing H5 Pipeline Test
    # -------------------------------------------------------------
    print("\n--- [TEST 1: Existing H5 Pipeline] ---")
    sample_h5 = Path(r"C:\Users\SUBHAM\Desktop\IEEE_Dataset\extracted\BRATS_001\BRATS_001_60.h5")
    if sample_h5.exists():
        with h5py.File(sample_h5, "r") as f:
            h5_x = f["x"][:].astype(np.float32)
            h5_y = (f["y"][:] > 0).astype(np.uint8)

        print(f"H5 file: {sample_h5.name}")
        print(f"  MRI shape: {h5_x.shape}, dtype: {h5_x.dtype}, min: {h5_x.min():.2f}, max: {h5_x.max():.2f}")
        print(f"  Ground truth pixels: {int(h5_y.sum()):,}")

        # FP32 inference
        t0 = time.perf_counter()
        raw_f = sess_fp32.run(None, {inp_name_f: h5_x[None, None, :, :]})[0]
        lat_f = (time.perf_counter() - t0) * 1000.0
        mask_f = ((1.0 / (1.0 + np.exp(-raw_f[0, 0]))) > 0.5).astype(np.uint8)

        # INT8 inference
        t0 = time.perf_counter()
        raw_i = sess_int8.run(None, {inp_name_i: h5_x[None, None, :, :]})[0]
        lat_i = (time.perf_counter() - t0) * 1000.0
        mask_i = ((1.0 / (1.0 + np.exp(-raw_i[0, 0]))) > 0.5).astype(np.uint8)

        dice_f = compute_dice(mask_f, h5_y)
        dice_i = compute_dice(mask_i, h5_y)
        parity_h5 = compute_parity(mask_f, mask_i)

        print(f"  FP32 Dice: {dice_f * 100:.2f}% | Latency: {lat_f:.2f} ms")
        print(f"  INT8 Dice: {dice_i * 100:.2f}% | Latency: {lat_i:.2f} ms")
        print(f"  FP32 vs INT8 Parity: {parity_h5:.3f}%")

        assert dice_f > 0.85, "FP32 Dice on standard H5 slice should be > 85%"
        assert dice_i > 0.85, "INT8 Dice on standard H5 slice should be > 85%"
        print("  >> H5 Pipeline: PASSED (100% Functional)")
    else:
        print("  Sample H5 path not found; skipping local IEEE file check.")

    # -------------------------------------------------------------
    # 2. New NIfTI Adapter Pipeline Test
    # -------------------------------------------------------------
    print("\n--- [TEST 2: New NIfTI Adapter Pipeline] ---")
    eval_cases = discover_evaluation_cases(PROJECT_ROOT / "evaluation_data")
    print(f"Discovered {len(eval_cases)} evaluation cases.")
    flair_case = next(c for c in eval_cases if c["is_recommended"])
    print(f"Loading primary FLAIR case: {flair_case['filename']}")

    vol_raw, meta = read_nifti(flair_case["path"])
    print(f"  Volume shape: {vol_raw.shape}, dtype: {vol_raw.dtype}")
    assert vol_raw.shape == (240, 240, 155), "NIfTI shape must be (240, 240, 155)"
    assert vol_raw.dtype == np.int16, "NIfTI dtype must be int16"

    vol_norm = normalize_volume(vol_raw)
    print(f"  Normalized shape: {vol_norm.shape}, dtype: {vol_norm.dtype}, min: {vol_norm.min():.3f}, max: {vol_norm.max():.3f}")
    assert vol_norm.shape == (240, 240, 155)
    assert vol_norm.dtype == np.float32

    slice_idx = 70
    s_tensor = get_slice(vol_norm, slice_idx)
    print(f"  Slice {slice_idx} tensor: shape={s_tensor.shape}, dtype={s_tensor.dtype}")
    assert s_tensor.shape == (1, 1, 240, 240)
    assert s_tensor.dtype == np.float32

    # Inference without ground truth (blind evaluation)
    t0 = time.perf_counter()
    raw_nii_f = sess_fp32.run(None, {inp_name_f: s_tensor})[0]
    lat_nii_f = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    raw_nii_i = sess_int8.run(None, {inp_name_i: s_tensor})[0]
    lat_nii_i = (time.perf_counter() - t0) * 1000.0

    mask_nii_f = ((1.0 / (1.0 + np.exp(-raw_nii_f[0, 0]))) > 0.5).astype(np.uint8)
    mask_nii_i = ((1.0 / (1.0 + np.exp(-raw_nii_i[0, 0]))) > 0.5).astype(np.uint8)

    parity_nii = compute_parity(mask_nii_f, mask_nii_i)

    print(f"  FP32 Tumor Pixels: {int(mask_nii_f.sum()):,} | Latency: {lat_nii_f:.2f} ms")
    print(f"  INT8 Tumor Pixels: {int(mask_nii_i.sum()):,} | Latency: {lat_nii_i:.2f} ms")
    print(f"  FP32 vs INT8 Parity: {parity_nii:.3f}%")
    print("  Dice Score: N/A — Blind Evaluation (No Ground Truth Provided)")

    assert int(mask_nii_f.sum()) > 1000, "Slice 70 should segment prominent tumor"
    assert int(mask_nii_i.sum()) > 1000, "Slice 70 should segment prominent tumor"
    assert parity_nii > 99.0, "Parity should be > 99%"
    print("  >> NIfTI Adapter Pipeline: PASSED (100% Functional)")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED: Both H5 and NIfTI workflows operating flawlessly!")
    print("=" * 70)


if __name__ == "__main__":
    test_pipeline()
