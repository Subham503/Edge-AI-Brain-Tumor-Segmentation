"""
NIfTI-1 Reader & Preprocessing Adapter for Edge AI Inference
=============================================================
Zero-dependency (Standard Library + NumPy) NIfTI reader designed for 
lightweight edge environments (Raspberry Pi, x86 CPU).

Meets IEEE hackathon requirements:
- No PyTorch / TensorFlow / Nibabel dependencies
- Reads uncompressed NIfTI-1 (.nii) files directly from disk or memory
- Whole-volume z-score normalization matching BraTS training distribution
- 2D axial slice extraction matching ONNX U-Net [1, 1, 240, 240] input
"""

import io
import os
from pathlib import Path
import struct
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# Datatype mapping according to NIfTI-1 specification
NIFTI_DATATYPES: Dict[int, Tuple[str, np.dtype, int]] = {
    2: ("UINT8", np.dtype(np.uint8), 1),
    4: ("INT16", np.dtype(np.int16), 2),
    8: ("INT32", np.dtype(np.int32), 4),
    16: ("FLOAT32", np.dtype(np.float32), 4),
    64: ("FLOAT64", np.dtype(np.float64), 8),
    512: ("UINT16", np.dtype(np.uint16), 2),
}


def read_nifti_header(source: Union[str, Path, io.BytesIO, Any]) -> Dict[str, Any]:
    """
    Parse and validate the 348-byte NIfTI-1 header from a file path or file-like object.
    
    Parameters
    ----------
    source : str, Path, or file-like object
        Path to .nii file or in-memory byte buffer.
        
    Returns
    -------
    dict
        Parsed metadata including dimensions, datatype, voxel spacing, and offset.
    """
    close_after = False
    if isinstance(source, (str, Path)):
        f = open(source, "rb")
        close_after = True
    elif hasattr(source, "read"):
        source.seek(0)
        f = source
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    try:
        hdr_bytes = f.read(352)
        if len(hdr_bytes) < 348:
            raise ValueError(f"File too short for NIfTI-1 header (got {len(hdr_bytes)} bytes, expected >= 348).")

        # Determine endianness from sizeof_hdr
        sizeof_hdr, = struct.unpack("<i", hdr_bytes[:4])
        endian = "<"
        if sizeof_hdr != 348:
            sizeof_hdr, = struct.unpack(">i", hdr_bytes[:4])
            endian = ">"
            if sizeof_hdr != 348:
                raise ValueError(f"Invalid NIfTI header size {sizeof_hdr}. Must be 348.")

        # Parse dimension array (dim[0] = ndim, dim[1..3] = nx, ny, nz)
        dim = struct.unpack(endian + "8h", hdr_bytes[40:56])
        ndim = dim[0]
        dimensions = tuple(dim[1:ndim + 1]) if ndim > 0 else ()

        # Datatype and bitpix
        datatype_code, bitpix = struct.unpack(endian + "hh", hdr_bytes[70:74])
        if datatype_code not in NIFTI_DATATYPES:
            raise ValueError(f"Unsupported NIfTI datatype code {datatype_code} (bitpix={bitpix}).")
        dtype_name, np_dtype, bytes_per_elem = NIFTI_DATATYPES[datatype_code]

        # Voxel spacing pixdim (pixdim[1..3] = dx, dy, dz)
        pixdim = struct.unpack(endian + "8f", hdr_bytes[76:108])
        voxel_spacing = tuple(pixdim[1:ndim + 1]) if ndim > 0 else ()

        # Byte offset to voxel data
        vox_offset, = struct.unpack(endian + "f", hdr_bytes[108:112])
        vox_offset = int(vox_offset)

        # Magic number
        magic = hdr_bytes[344:348].decode("latin1", errors="replace")
        if not (magic.startswith("n+1") or magic.startswith("ni1")):
            raise ValueError(f"Invalid NIfTI magic number '{magic}'. Must be 'n+1' or 'ni1'.")

        # Description string (up to 80 chars)
        descrip = hdr_bytes[148:228].split(b"\x00")[0].decode("latin1", errors="replace")

        return {
            "endian": endian,
            "sizeof_hdr": sizeof_hdr,
            "ndim": ndim,
            "dimensions": dimensions,
            "datatype_code": datatype_code,
            "datatype_name": dtype_name,
            "np_dtype": np_dtype,
            "bitpix": bitpix,
            "bytes_per_elem": bytes_per_elem,
            "voxel_spacing": voxel_spacing,
            "vox_offset": vox_offset,
            "magic": magic,
            "descrip": descrip,
        }
    finally:
        if close_after:
            f.close()


def read_nifti_volume(source: Union[str, Path, io.BytesIO, Any]) -> np.ndarray:
    """
    Read an uncompressed NIfTI-1 3D volume into a NumPy array.
    
    Validates that the file conforms to standard BraTS dimensions:
    approx (240, 240, 155), INT16 datatype.
    
    Parameters
    ----------
    source : str, Path, or file-like object
        Path to .nii file or buffer.
        
    Returns
    -------
    np.ndarray
        3D NumPy array of shape (240, 240, 155) and dtype np.int16.
    """
    volume, _ = read_nifti(source)
    return volume


def read_nifti(
    source: Union[str, Path, io.BytesIO, Any]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read NIfTI volume and metadata.
    
    Parameters
    ----------
    source : str, Path, or file-like object
        Path or byte buffer.
        
    Returns
    -------
    volume : np.ndarray
        Array with shape (240, 240, 155), dtype matching header (np.int16).
    meta : dict
        Header metadata dictionary.
    """
    meta = read_nifti_header(source)
    dimensions = meta["dimensions"]
    np_dtype = meta["np_dtype"]
    vox_offset = meta["vox_offset"]
    endian = meta["endian"]

    # Validate dimensions (BraTS standard: 240x240x155)
    if len(dimensions) < 3:
        raise ValueError(f"Expected at least 3 dimensions for 3D MRI volume, got {dimensions}.")
    
    shape_3d = (dimensions[0], dimensions[1], dimensions[2])

    close_after = False
    if isinstance(source, (str, Path)):
        f = open(source, "rb")
        close_after = True
    else:
        source.seek(0)
        f = source

    try:
        f.seek(vox_offset)
        raw_bytes = f.read()
        
        expected_bytes = int(shape_3d[0] * shape_3d[1] * shape_3d[2] * meta["bytes_per_elem"])
        if len(raw_bytes) < expected_bytes:
            raise ValueError(
                f"Voxel data incomplete: read {len(raw_bytes)} bytes, expected {expected_bytes} bytes."
            )

        # Apply endianness to dtype if needed
        dtype_with_endian = np_dtype.newbyteorder(endian)
        
        # In NIfTI, column-major storage order: X varies fastest, then Y, then Z.
        # Order 'F' produces array indexed as [X, Y, Z] == (240, 240, 155).
        volume = np.frombuffer(raw_bytes[:expected_bytes], dtype=dtype_with_endian).reshape(
            shape_3d, order="F"
        )
        
        # Ensure native byte order in memory
        if volume.dtype.byteorder not in ("=", "|"):
            volume = volume.astype(volume.dtype.newbyteorder("="))

        return volume, meta
    finally:
        if close_after:
            f.close()


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """
    Apply whole-volume z-score intensity normalization.
    
    Matches the exact training distribution used for the BraTS U-Net:
        normalized = (volume - mean) / (std + 1e-8)
        
    Parameters
    ----------
    volume : np.ndarray
        Raw 3D MRI volume (e.g., INT16).
        
    Returns
    -------
    np.ndarray
        Float32 normalized volume with zero mean and unit variance across voxels.
    """
    vol_f32 = volume.astype(np.float32)
    mean = float(vol_f32.mean())
    std = float(vol_f32.std())
    
    normalized = (vol_f32 - mean) / (std + 1e-8)
    return normalized


def get_slice(volume: np.ndarray, slice_index: int) -> np.ndarray:
    """
    Extract a 2D axial slice formatted for ONNX U-Net input contract.
    
    Parameters
    ----------
    volume : np.ndarray
        Normalized or raw 3D volume with shape (240, 240, 155).
    slice_index : int
        Axial slice index (0 to 154).
        
    Returns
    -------
    np.ndarray
        4D tensor with shape [1, 1, 240, 240] and dtype np.float32.
    """
    if slice_index < 0 or slice_index >= volume.shape[2]:
        raise IndexError(
            f"Slice index {slice_index} out of bounds for volume with {volume.shape[2]} slices."
        )

    slice_2d = volume[:, :, slice_index].astype(np.float32)
    return slice_2d[None, None, :, :]  # Shape: [1, 1, 240, 240]


def get_slice_2d(volume: np.ndarray, slice_index: int) -> np.ndarray:
    """
    Extract a single 2D axial slice array of shape (240, 240) as float32.
    Convenience method for plotting, overlays, and edge metrics.
    """
    if slice_index < 0 or slice_index >= volume.shape[2]:
        raise IndexError(
            f"Slice index {slice_index} out of bounds for volume with {volume.shape[2]} slices."
        )

    return volume[:, :, slice_index].astype(np.float32)


def discover_evaluation_cases(
    eval_dir: Union[str, Path]
) -> List[Dict[str, Any]]:
    """
    Discover all available NIfTI test cases within evaluation_data.
    
    Handles:
    - Normal subdirectories
    - Subdirectories named ending with .nii (unpacked zip artifacts)
    - 8.3 shortname directories like BRATS2~2.ZI0
    
    Returns
    -------
    list of dict
        Discovered cases sorted with FLAIR prioritized first.
        Each item has:
        - "path": Path to the .nii file
        - "filename": Name of file
        - "modality": Detected modality ('FLAIR', 'T1', 'T1ce', 'T2', 'UNKNOWN')
        - "case_id": Detected patient/case ID (e.g. 'BraTS20_Training_030')
        - "is_recommended": True if FLAIR (matches U-Net training)
        - "filesize_mb": Size in megabytes
    """
    eval_path = Path(eval_dir)
    if not eval_path.exists():
        return []

    discovered = []
    # Find all .nii files recursively
    for file_path in eval_path.rglob("*.nii"):
        if not file_path.is_file():
            continue

        fname_lower = file_path.name.lower()
        
        # Detect modality
        if "flair" in fname_lower:
            modality = "FLAIR"
            is_recommended = True
        elif "t1ce" in fname_lower:
            modality = "T1ce"
            is_recommended = False
        elif "t1" in fname_lower:
            modality = "T1"
            is_recommended = False
        elif "t2" in fname_lower:
            modality = "T2"
            is_recommended = False
        else:
            modality = "UNKNOWN"
            is_recommended = False

        # Extract case ID
        parts = file_path.stem.split("_")
        if len(parts) >= 3 and parts[0].lower().startswith("brats"):
            case_id = "_".join(parts[:3])  # e.g., BraTS20_Training_030
        else:
            case_id = file_path.stem

        size_mb = file_path.stat().st_size / (1024 * 1024)

        discovered.append({
            "path": file_path,
            "filename": file_path.name,
            "modality": modality,
            "case_id": case_id,
            "is_recommended": is_recommended,
            "filesize_mb": round(size_mb, 2),
        })

    # Sort with recommended FLAIR first, then alphabetically
    discovered.sort(key=lambda x: (not x["is_recommended"], x["modality"], x["filename"]))
    return discovered
