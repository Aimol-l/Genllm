#!/usr/bin/env python3
"""
Genllm unified build script.

Usage:
    python build.py                          # CPU only (default)
    python build.py --vulkan                 # CPU + Vulkan
    python build.py --cuda                   # CPU + CUDA
    python build.py --sycl                   # CPU + SYCL
    python build.py --test                   # also build tests
    python build.py --shader                 # only compile shaders -> SPIR-V -> headers
    python build.py --clean                  # clean everything
    python build.py --rebuild                # clean + build
    python build.py -j8                      # parallel build with 8 jobs
"""

import argparse
import shutil
import struct
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT   = Path(__file__).resolve().parent
BUILD_DIR      = PROJECT_ROOT / "build"
SHADER_DIR     = PROJECT_ROOT / "shader"
SPV_DIR        = BUILD_DIR / "spv"
SPV_HEADER_DIR = PROJECT_ROOT / "include" / "backend" / "vulkan" / "spv"

GLSLC = "glslc"


# ======================== Shader Compilation ========================

def compile_shader(comp_path: Path, spv_path: Path) -> bool:
    spv_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [GLSLC, str(comp_path), "-o", str(spv_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  FAIL {comp_path.name}: {result.stderr.strip()}")
        return False
    return True


def spv_to_words(spv_path: Path) -> tuple[list[int], int]:
    data = spv_path.read_bytes()
    n = len(data) // 4
    words = list(struct.unpack(f"<{n}I", data))
    return words, n


def generate_spv_header(op_name: str, variants: dict[str, tuple[list[int], int]]) -> Path:
    lines = [
        "#pragma once",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace vkspv {",
        "",
    ]
    for vname in sorted(variants):
        _, n = variants[vname]
        lines.append(f"inline constexpr size_t {vname}_spv_len = {n};")
    lines.append("")
    for vname in sorted(variants):
        words, n = variants[vname]
        lines.append(f"inline constexpr uint32_t {vname}_spv[] = {{")
        for i in range(0, n, 8):
            chunk = words[i : i + 8]
            row = "    " + ", ".join(f"0x{w:08x}" for w in chunk)
            if i + 8 < n:
                row += ","
            lines.append(row)
        lines.append("};")
        lines.append("")
    lines.append("} // namespace vkspv")
    lines.append("")

    header_path = SPV_HEADER_DIR / f"{op_name}.h"
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text("\n".join(lines))
    return header_path


def build_shaders() -> bool:
    if not SHADER_DIR.exists():
        print("No shader directory found.")
        return True

    ops: dict[str, dict[str, Path]] = {}
    for comp_file in sorted(SHADER_DIR.rglob("*.comp")):
        ops.setdefault(comp_file.parent.name, {})[comp_file.stem] = comp_file

    if not ops:
        print("No shaders found in", SHADER_DIR)
        return True

    SPV_DIR.mkdir(parents=True, exist_ok=True)
    total = sum(len(v) for v in ops.values())
    print(f"\n[Shader] Compiling {total} shaders ({len(ops)} ops) ...")

    errors = 0
    for op_name, variants in sorted(ops.items()):
        compiled: dict[str, tuple[list[int], int]] = {}
        for vname, comp_path in sorted(variants.items()):
            spv_path = SPV_DIR / f"{vname}.spv"
            rel = comp_path.relative_to(PROJECT_ROOT)
            print(f"  {rel} ", end="", flush=True)

            if not compile_shader(comp_path, spv_path):
                errors += 1
                continue

            words, n = spv_to_words(spv_path)
            compiled[vname] = (words, n)
            kb = len(spv_path.read_bytes()) / 1024
            print(f"-> {spv_path.name}  ({n} words, {kb:.1f} KB)")

        if compiled:
            header = generate_spv_header(op_name, compiled)
            print(f"  => {header.relative_to(PROJECT_ROOT)}  ({len(compiled)} variants)")

    if errors:
        print(f"\n[Shader] {errors} shader(s) failed!")
        return False
    print(f"[Shader] Done. {total - errors}/{total} compiled.")
    return True


def clean_shaders():
    if SPV_DIR.exists():
        shutil.rmtree(SPV_DIR)
        print(f"Removed {SPV_DIR}")
    if SPV_HEADER_DIR.exists():
        for f in SPV_HEADER_DIR.glob("*.h"):
            f.unlink()
            print(f"Removed {f}")


# ======================== CMake Build ========================

def cmake_configure(backends: dict[str, bool], build_test: bool) -> bool:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "cmake", str(PROJECT_ROOT),
        "-G", "Ninja" if shutil.which("ninja") else "Unix Makefiles",
    ]
    if shutil.which("ninja"):
        cmd += ["-G", "Ninja"]

    for name, enabled in backends.items():
        flag = f"-DBACKEND_{name.upper()}={'ON' if enabled else 'OFF'}"
        cmd.append(flag)

    cmd.append(f"-DBUILD_TEST={'ON' if build_test else 'OFF'}")

    print(f"\n[CMake] Configure: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=BUILD_DIR)
    if result.returncode != 0:
        print("[CMake] Configure failed!")
        return False
    print("[CMake] Configure done.")
    return True


def cmake_build(jobs: int) -> bool:
    cmd = ["cmake", "--build", str(BUILD_DIR)]
    if jobs > 1:
        cmd.append(f"-j{jobs}")

    print(f"\n[Build] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[Build] Failed!")
        return False
    print("[Build] Done.")
    return True


def clean_cmake():
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
        print(f"Removed {BUILD_DIR}")


# ======================== Main ========================

def main():
    parser = argparse.ArgumentParser(
        description="Genllm unified build script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python build.py                      # CPU only (default)
    python build.py --vulkan             # CPU + Vulkan
    python build.py --cuda               # CPU + CUDA
    python build.py --sycl               # CPU + SYCL
    python build.py --shader             # compile shaders only
    python build.py --rebuild -j$(nproc) # full rebuild with max parallelism
        """,
    )
    parser.add_argument("--cpu",     action="store_true", help="CPU only (no GPU backend)")
    parser.add_argument("--cuda",    action="store_true", help="CPU + CUDA backend")
    parser.add_argument("--vulkan",  action="store_true", help="CPU + Vulkan backend")
    parser.add_argument("--sycl",    action="store_true", help="CPU + SYCL backend")
    parser.add_argument("--test",    action="store_true", help="build tests")
    parser.add_argument("--shader",  action="store_true", help="only compile shaders (no cmake)")
    parser.add_argument("--clean",   action="store_true", help="remove build artifacts")
    parser.add_argument("--rebuild", action="store_true", help="clean + build")
    parser.add_argument("-j", type=int, default=0, help="parallel build jobs (default: auto)")
    args = parser.parse_args()

    # --clean or --rebuild
    if args.clean:
        clean_shaders()
        clean_cmake()
        return

    if args.rebuild:
        clean_shaders()
        clean_cmake()

    # --shader only
    if args.shader:
        ok = build_shaders()
        sys.exit(0 if ok else 1)

    # Determine backends: CPU is always enabled; pick at most one GPU backend
    gpu_flags = [("cuda", args.cuda), ("vulkan", args.vulkan), ("sycl", args.sycl)]
    gpu_enabled = [name for name, on in gpu_flags if on]
    if len(gpu_enabled) > 1:
        parser.error(f"Only one GPU backend allowed, got: {', '.join(gpu_enabled)}")

    backends = {"cpu": True}
    if gpu_enabled:
        backends[gpu_enabled[0]] = True
    elif not args.cpu:
        # Default: CPU only
        print("[Info] No backend specified, defaulting to CPU only.")

    # Shader step (needed for Vulkan)
    if backends["vulkan"]:
        if not build_shaders():
            sys.exit(1)

    # CMake steps
    jobs = args.j if args.j > 0 else (0)  # 0 = let cmake decide

    if not cmake_configure(backends, args.test):
        sys.exit(1)
    if not cmake_build(jobs):
        sys.exit(1)

    print("\nAll done.")


if __name__ == "__main__":
    main()
