"""
diagnose_h5_files.py

Diagnostic tool to inspect structural hierarchy, dataset shapes, data types,
and attribute metadata inside the pipeline HDF5 files:
  1. processed_experimental_data.h5
  2. tlv_fit_results.h5
  3. eda_results.h5

Saves the full detailed diagnostic report to a text file instead of printing
excessive output to the terminal.
"""

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

# Attempt to load project configuration for default paths
try:
    from creep_model.config import config
    DATA_DIR = Path(config.data_output_directory)
except ImportError:
    DATA_DIR = Path("data/processed")

# Target HDF5 files to diagnose
H5_FILES = [
    "processed_experimental_data.h5",
    "tlv_fit_results.h5",
    "eda_results.h5",
]

# Destination for the diagnostic report
REPORT_PATH = DATA_DIR / "h5_diagnostics_report.txt" if DATA_DIR.exists() else Path(tempfile.gettempdir()) / "h5_diagnostics_report.txt"


def format_value(val):
    """Formats attribute or scalar values cleanly for file display."""
    if isinstance(val, (bytes, bytearray)):
        return val.decode("utf-8")
    if isinstance(val, np.ndarray):
        if val.size <= 5:
            return str(val.tolist())
        return f"Array(shape={val.shape}, dtype={val.dtype})"
    if isinstance(val, float):
        return f"{val:.6e}" if (abs(val) < 1e-3 or abs(val) > 1e4) and val != 0 else f"{val:.4f}"
    return str(val)


def print_attributes(attrs, f_out, indent=""):
    """Prints all metadata attributes associated with a group or dataset to file."""
    if not attrs:
        return
    f_out.write(f"{indent}  Attributes:\n")
    for k, v in attrs.items():
        f_out.write(f"{indent}    - {k}: {format_value(v)}\n")


def inspect_node(name, obj, f_out, indent=""):
    """Callback function for h5py.File.visititems to write hierarchy details."""
    depth = name.count("/")
    current_indent = "  " * (depth + 1)
    node_name = name.split("/")[-1]

    if isinstance(obj, h5py.Group):
        f_out.write(f"\n{current_indent}[GROUP] /{name}\n")
        print_attributes(obj.attrs, f_out, current_indent)

    elif isinstance(obj, h5py.Dataset):
        dtype_str = str(obj.dtype)
        shape_str = str(obj.shape)
        
        try:
            arr = obj[:]
            if np.issubdtype(arr.dtype, np.number) and arr.size > 0:
                valid_mask = ~np.isnan(arr)
                if np.any(valid_mask):
                    v_min = np.min(arr[valid_mask])
                    v_max = np.max(arr[valid_mask])
                    v_mean = np.mean(arr[valid_mask])
                    summary = f"min={v_min:.4e}, max={v_max:.4e}, mean={v_mean:.4e}"
                else:
                    summary = "all NaN"
            elif arr.dtype.kind in ['S', 'U', 'O']:
                sample = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in arr[:3]]
                summary = f"Sample strings: {sample}"
            else:
                summary = f"size={arr.size}"
        except Exception as e:
            summary = f"Error reading data: {e}"

        f_out.write(f"{current_indent}[DATASET] {node_name} | Shape: {shape_str} | Type: {dtype_str} | Summary: {summary}\n")
        print_attributes(obj.attrs, f_out, current_indent)


def inspect_h5_file(file_path: Path, f_out):
    """Inspects a single HDF5 file and writes structure details to the report file."""
    f_out.write("=" * 80 + "\n")
    f_out.write(f" DIAGNOSING FILE: {file_path.name}\n")
    f_out.write(f" Absolute Path : {file_path.absolute()}\n")
    f_out.write("=" * 80 + "\n")

    if not file_path.exists():
        f_out.write(f"  [ERROR] File does not exist at {file_path.absolute()}\n\n")
        return

    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    f_out.write(f" File Size: {file_size_mb:.2f} MB\n")

    try:
        with h5py.File(file_path, "r") as f:
            f_out.write("\n Root Group Attributes:\n")
            if f.attrs:
                for k, v in f.attrs.items():
                    f_out.write(f"   - {k}: {format_value(v)}\n")
            else:
                f_out.write("   (None)\n")

            f_out.write("\n Structure Tree:\n")
            f.visititems(lambda name, obj: inspect_node(name, obj, f_out))

            # Data Integrity Check
            f_out.write("\n Data Integrity Diagnostics:\n")
            flags = []
            
            def check_integrity(name, obj):
                if isinstance(obj, h5py.Group):
                    for k, v in obj.attrs.items():
                        if isinstance(v, (int, float)) and v == 0.0:
                            flags.append(f"Group '{name}' attribute '{k}' == 0.0")
                elif isinstance(obj, h5py.Dataset):
                    if np.issubdtype(obj.dtype, np.number) and obj.size > 0:
                        arr = obj[:]
                        if np.all(arr == 0.0):
                            flags.append(f"Dataset '{name}' is ALL ZEROS (shape={obj.shape})")
                        elif np.all(arr == arr[0]):
                            flags.append(f"Dataset '{name}' has CONSTANT values ({arr[0]})")

            f.visititems(check_integrity)

            if flags:
                f_out.write("   Found potential missing/constant data issues:\n")
                for flag in flags[:20]:
                    f_out.write(f"   [!] {flag}\n")
                if len(flags) > 20:
                    f_out.write(f"   ... and {len(flags) - 20} more constant/zero fields.\n")
            else:
                f_out.write("   [OK] No all-zero or constant numerical datasets found.\n")

    except Exception as err:
        f_out.write(f"  [ERROR] Failed to read HDF5 file: {err}\n")

    f_out.write("\n\n")


def main():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    search_directories = [
        DATA_DIR,
        Path.cwd() / "data" / "processed",
        Path(__file__).resolve().parent / "data" / "processed",
        Path.cwd(),
    ]

    with open(REPORT_PATH, "w", encoding="utf-8") as f_out:
        f_out.write("#" * 80 + "\n")
        f_out.write(" HDF5 PIPELINE DATA DIAGNOSTIC REPORT\n")
        f_out.write("#" * 80 + "\n\n")

        for file_name in H5_FILES:
            target_path = None
            for s_dir in search_directories:
                candidate = s_dir / file_name
                if candidate.exists():
                    target_path = candidate
                    break

            if target_path:
                inspect_h5_file(target_path, f_out)
            else:
                f_out.write("=" * 80 + "\n")
                f_out.write(f" DIAGNOSING FILE: {file_name}\n")
                f_out.write("=" * 80 + "\n")
                f_out.write(f"  [ERROR] File '{file_name}' not found in search paths.\n\n")

    print(f"\n[SUCCESS] Diagnostic report successfully saved to:\n  -> {REPORT_PATH.absolute()}\n")


if __name__ == "__main__":
    main()