"""Phase 0: environment audit. Prints and writes report/env_report.md (environment section)."""
from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-c", "(Get-CimInstance Win32_Processor).Name"], text=True
            )
            return out.strip()
        except Exception:
            pass
    elif platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or "unknown"


def _ram_gb() -> tuple[float | None, float | None]:
    try:
        import psutil  # optional

        vm = psutil.virtual_memory()
        return vm.total / 1e9, vm.available / 1e9
    except ImportError:
        pass
    if platform.system() == "Windows":
        import ctypes

        class MEMSTAT(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        m = MEMSTAT()
        m.dwLength = ctypes.sizeof(MEMSTAT)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys / 1e9, m.ullAvailPhys / 1e9
    if hasattr(os, "sysconf"):
        pages, size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
        avail = os.sysconf("SC_AVPHYS_PAGES") * size
        return pages * size / 1e9, avail / 1e9
    return None, None


def collect() -> dict:
    total, avail = _ram_gb()
    info = {
        "OS": platform.platform(),
        "CPU": _cpu_name(),
        "Logical cores": os.cpu_count(),
        "RAM total (GB)": f"{total:.1f}" if total else "unknown",
        "RAM available (GB)": f"{avail:.1f}" if avail else "unknown",
        "Free disk (GB)": f"{shutil.disk_usage(ROOT).free / 1e9:.0f}",
        "Python": sys.version.split()[0],
        "Python executable": sys.executable,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["CUDA available"] = torch.cuda.is_available()
        info["MPS available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["GPU"] = p.name
            info["VRAM (GB)"] = f"{p.total_memory / 1e9:.1f}"
            info["Compute capability"] = f"{p.major}.{p.minor}"
            info["CUDA (torch build)"] = torch.version.cuda
            info["cuDNN"] = torch.backends.cudnn.version()
    except ImportError:
        info["torch"] = "NOT INSTALLED"
    try:
        drv = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip()
        info["NVIDIA driver"] = drv
    except Exception:
        info["NVIDIA driver"] = "n/a"
    for pkg in ["numpy", "scipy", "matplotlib", "yaml", "pandas", "pytest", "neuralop"]:
        try:
            m = importlib.import_module(pkg)
            info[f"pkg:{pkg}"] = getattr(m, "__version__", "installed")
        except ImportError:
            info[f"pkg:{pkg}"] = "NOT INSTALLED"
    return info


def main() -> None:
    info = collect()
    lines = ["# Environment report", "", "## Hardware and software", "", "| Item | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in info.items()]
    text = "\n".join(lines) + "\n"
    print(text)
    out = ROOT / "report" / "env_report.md"
    out.parent.mkdir(exist_ok=True)
    # Preserve any benchmark section appended by 01_benchmark.py
    marker = "## Benchmarks"
    if out.exists() and marker in out.read_text():
        text += "\n" + out.read_text()[out.read_text().index(marker):]
    out.write_text(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
