# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build isolated, pinned MXFP4 CPU libraries for the hybrid MoE adapter."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

SOURCES = {
    "llama": ("ggml-org/llama.cpp", "c069aa7f5f2beeead1a3a8e9f71510f1b64d0725"),
    "ik": ("ikawrakow/ik_llama.cpp", "3bb386eb68ffee0a5dc7db21da0735d594929eeb"),
    "kt": ("kvcache-ai/ktransformers", "440df948027128318bc30a11a545433eb0c7d383"),
}
KT_SUBMODULES = {
    "third_party/llama.cpp": (
        "ggml-org/llama.cpp",
        "a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f",
    ),
    "third_party/pybind11": (
        "pybind/pybind11",
        "bb05e0810b87e74709d9f4c4545f1f57a1b386f5",
    ),
}


def run(command, **kwargs):
    print("Running:", command, flush=True)
    subprocess.run([str(arg) for arg in command], check=True, **kwargs)


def verify_source(destination, repository, revision):
    marker = destination / ".dsv41-source.json"
    if marker.is_file():
        recorded = json.loads(marker.read_text())
        if (recorded["repository"], recorded["revision"]) == (repository, revision):
            return
    if (destination / ".git").exists():
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=destination, text=True
        ).strip()
        if actual == revision:
            return
    raise RuntimeError(
        f"Cannot verify pinned source at {destination}; "
        "use a fresh work directory with --download"
    )


def download(destination, repository, revision):
    if destination.is_dir() and any(destination.iterdir()):
        verify_source(destination, repository, revision)
        return
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination.parent / f"{destination.name}-{revision}.tar.gz"
    urllib.request.urlretrieve(
        f"https://codeload.github.com/{repository}/tar.gz/{revision}", archive
    )
    with tarfile.open(archive) as contents:
        for member in contents:
            _, _, name = member.name.partition("/")
            if name:
                member.name = name
                contents.extract(member, destination, filter="data")
    (destination / ".dsv41-source.json").write_text(
        json.dumps(
            {
                "repository": repository,
                "revision": revision,
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--backends", nargs="*", choices=SOURCES, default=list(SOURCES))
    parser.add_argument("--cuda-bridge", action="store_true")
    parser.add_argument("--cuda-home", type=Path, default=Path("/usr/local/cuda"))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--install-dir", type=Path)
    args = parser.parse_args()
    root = args.work_dir.resolve()
    source_root, build_root = root / "sources", root / "build"
    build_root.mkdir(parents=True, exist_ok=True)
    wrapper_root = Path(__file__).resolve().parent
    patch = wrapper_root / "ik-pre-silu-clamp.patch"
    manifest = {"sources": SOURCES, "kt_submodules": KT_SUBMODULES, "libraries": {}}
    manifest["ik_patch_sha256"] = hashlib.sha256(patch.read_bytes()).hexdigest()
    manifest["wrapper_sha256"] = hashlib.sha256(
        (wrapper_root / "ggml_moe.cpp").read_bytes()
    ).hexdigest()
    manifest["compact_profile_sha256"] = hashlib.sha256(
        (wrapper_root / "compact_profile.h").read_bytes()
    ).hexdigest()
    manifest["compact_executor_sha256"] = hashlib.sha256(
        (wrapper_root / "compact_moe.h").read_bytes()
    ).hexdigest()
    manifest["compact_avx2_sha256"] = hashlib.sha256(
        (wrapper_root / "compact_avx2.h").read_bytes()
    ).hexdigest()
    manifest["numa_executor_sha256"] = hashlib.sha256(
        (wrapper_root / "numa.h").read_bytes()
    ).hexdigest()
    for name in args.backends:
        source, build = source_root / name, build_root / name
        if args.download:
            download(source, *SOURCES[name])
            if name == "kt":
                for relative, origin in KT_SUBMODULES.items():
                    download(source / relative, *origin)
        if not source.is_dir():
            parser.error(f"Missing {source}; use --download for the pinned snapshots")
        verify_source(source, *SOURCES[name])
        if name == "kt":
            for relative, origin in KT_SUBMODULES.items():
                verify_source(source / relative, *origin)
            env = os.environ | {
                "CPUINFER_CPU_INSTRUCT": "AVX2",
                "CPUINFER_USE_CUDA": "1",
                "CPUINFER_ENABLE_AMX": "OFF",
                "CPUINFER_ENABLE_AVX512": "OFF",
                "CPUINFER_ENABLE_ONEDNN_VNNI": "OFF",
                "CPUINFER_PARALLEL": str(args.jobs),
            }
            run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                cwd=source / "kt-kernel",
                env=env,
            )
            libraries = list((source / "kt-kernel/python").glob("kt_kernel_ext*.so"))
            if len(libraries) != 1:
                raise RuntimeError(f"Expected one KT extension, got {libraries}")
            library = libraries[0]
        else:
            options = [
                "-DBUILD_SHARED_LIBS=OFF",
                "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
                "-DGGML_NATIVE=ON",
                "-DGGML_CUDA=OFF",
                "-DLLAMA_BUILD_TESTS=OFF",
                "-DLLAMA_BUILD_EXAMPLES=OFF",
            ]
            if name == "ik":
                check = subprocess.run(
                    ["git", "apply", "--reverse", "--check", str(patch)],
                    cwd=source,
                    capture_output=True,
                )
                if check.returncode:
                    run(["git", "apply", "--check", patch], cwd=source)
                    run(["git", "apply", patch], cwd=source)
                options += [
                    "-DGGML_IQK_MUL_MAT=ON",
                    "-DLLAMA_BUILD_SERVER=OFF",
                    "-DGGML_IQK_FLASH_ATTENTION=ON",
                    "-DGGML_IQK_FA_ALL_QUANTS=OFF",
                ]
            else:
                options += [
                    "-DGGML_CPU_REPACK=ON",
                    "-DLLAMA_BUILD_TOOLS=OFF",
                    "-DLLAMA_CURL=OFF",
                ]
            run(["cmake", "-S", source, "-B", build, *options])
            run(["cmake", "--build", build, "--target", "ggml", "-j", args.jobs])
            archives = [build / "ggml/src/libggml.a"]
            if name == "llama":
                archives += [build / f"ggml/src/libggml-{p}.a" for p in ("cpu", "base")]
            library = build_root / f"libdsv41_{name}.so"
            run(
                [
                    os.environ.get("CXX", "c++"),
                    "-std=c++17",
                    "-O3",
                    "-march=native",
                    "-fPIC",
                    "-shared",
                    wrapper_root / "ggml_moe.cpp",
                    *(["-DDSV41_IK"] if name == "ik" else []),
                    f"-I{source}/ggml/include",
                    f"-I{source}/ggml/src",
                    "-Wl,--start-group",
                    *archives,
                    "-Wl,--end-group",
                    "-Wl,--exclude-libs,ALL",
                    "-Wl,-Bsymbolic",
                    "-fopenmp",
                    "-pthread",
                    "-ldl",
                    "-o",
                    library,
                ]
            )
        manifest["libraries"][name] = {
            "path": str(library),
            "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        }
    if args.cuda_bridge:
        source = wrapper_root / "cuda_host_moe.cpp"
        library = build_root / "libdsv41_cuda.so"
        cuda_home = args.cuda_home.resolve()
        run(
            [
                os.environ.get("CXX", "c++"),
                "-std=c++17",
                "-O3",
                "-fPIC",
                "-shared",
                f"-I{cuda_home}/include",
                source,
                f"-L{cuda_home}/lib64",
                f"-Wl,-rpath,{cuda_home}/lib64",
                "-lcudart",
                "-o",
                library,
            ]
        )
        manifest["libraries"]["cuda_bridge"] = {
            "path": str(library),
            "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "cuda_home": str(cuda_home),
        }
    (build_root / "cpu-backends.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.install_dir:
        args.install_dir.mkdir(parents=True, exist_ok=True)
        for item in manifest["libraries"].values():
            source = Path(item["path"])
            shutil.copy2(source, args.install_dir / source.name)


if __name__ == "__main__":
    main()
