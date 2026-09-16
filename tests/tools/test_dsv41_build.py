# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build contracts for the ctypes libraries used by DeepSeek hybrid serving."""

import ast
import os
import shutil
import zipfile
from pathlib import Path

import pytest
import regex as re
from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext


@pytest.fixture
def build_classes():
    # Load the command classes without running setup() or downloading a wheel.
    source = Path(__file__).resolve().parents[2] / "setup.py"
    names = {"cmake_build_ext", "precompiled_build_ext", "precompiled_wheel_utils"}
    tree = ast.parse(source.read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in names
    ]
    namespace = {
        "os": os,
        "re": re,
        "shutil": shutil,
        "Path": Path,
        "build_ext": build_ext,
        "CMakeExtension": Extension,
    }
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("command", ["cmake_build_ext", "precompiled_build_ext"])
def test_ctypes_libraries_keep_their_names_in_editable_and_wheel_builds(
    build_classes, command, tmp_path
):
    """setuptools must track the same .so names that CMake and ctypes use."""
    extensions = [Extension(f"vllm.libdsv41_{name}", []) for name in ("ik", "cuda")]
    dist = Distribution({"ext_modules": extensions})
    build = build_classes[command](dist)
    build.ensure_finalized()
    build.build_lib = str(tmp_path)
    assert {Path(path).name for path in build.get_outputs()} == {
        "libdsv41_ik.so",
        "libdsv41_cuda.so",
    }
    build.inplace = True
    assert Path(build.get_ext_fullpath("vllm.libdsv41_ik")).name == "libdsv41_ik.so"
    assert "abi3" not in build.get_ext_filename("vllm.libdsv41_cuda")
    assert build.get_ext_filename("vllm._C") != "vllm/_C.so"


def test_precompiled_branch_wheel_extracts_hybrid_libraries_and_license(
    build_classes, tmp_path, monkeypatch
):
    """Python-only installs must retain the branch's native runtime dependencies."""
    members = {
        "vllm/libdsv41_ik.so": b"ik library",
        "vllm/libdsv41_cuda.so": b"cuda bridge",
        "vllm/third_party/ik_llama/LICENSE": b"MIT License",
    }
    wheel = tmp_path / "branch.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        archive.writestr("vllm/models/unrelated.py", b"do not overwrite")
    monkeypatch.chdir(tmp_path)
    packages = build_classes[
        "precompiled_wheel_utils"
    ].extract_precompiled_and_patch_package(
        str(wheel), None, extract_extensions=True, extract_rust_frontend=False
    )
    for name, data in members.items():
        assert (tmp_path / name).read_bytes() == data
    assert set(packages["vllm"]) == {"libdsv41_ik.so", "libdsv41_cuda.so"}
    assert not (tmp_path / "vllm/models/unrelated.py").exists()
