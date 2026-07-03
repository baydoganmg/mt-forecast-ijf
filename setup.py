# Build Cython extension(s):
#   python setup.py build_ext --inplace
#
# Builds BOTH fp64 (default) and fp32 versions of the multi-target tree kernels.
# Runtime selection is via the MT_DTYPE environment variable in mttrees/tree.py:
#   MT_DTYPE=fp64  -> mttrees.cfuncs.mt_ctree_{prebin,selfhash}        (default)
#   MT_DTYPE=fp32  -> mttrees.cfuncs.mt_ctree_{prebin,selfhash}_f32
#
# fp32 builds use the same kernel sources with `double` -> `float` substitution.
# They cut the data-matrix memory by ~50% at the cost of ~0.5% MASE drift on
# ensembles and bit-close DT predictions.
#
# Compile flags:
#   -O3 -march=native -mfma -mprefer-vector-width=256
#     aggressive optimisation + AVX2/FMA, but avoids AVX-512 downclock on
#     hybrid Intel CPUs (P-core + E-core mix).
#   -ffast-math -funroll-loops -ftree-vectorize
#     allow associative FP reordering, unroll inner loops, force vectoriser
#     passes. Drift vs strict-IEEE is acceptable for tree split scoring.
#   -flto
#     link-time optimisation. Enables cross-translation-unit inlining,
#     often 5-15% extra speed on numeric kernels.
#   -fopenmp
#     OpenMP for cython.parallel.prange.

import os
import platform
import shutil
import subprocess
import sys

import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

# Path B: pybind11 setup helpers for the cfuncs_cpp backend.
try:
    from pybind11.setup_helpers import Pybind11Extension
    _HAVE_PYBIND11 = True
except ImportError:
    _HAVE_PYBIND11 = False


def _platform_flags():
    """Return arch-aware compile/link flags.

    Linux x86_64 / Apple Silicon / Intel Mac / Windows handled individually.
    On macOS we look for libomp via `brew --prefix libomp` first, then fall
    back to the standard /opt/homebrew (Apple Silicon) or /usr/local (Intel
    Mac) install paths. If libomp is not found the build proceeds without
    OpenMP (kernel runs single-threaded).
    """
    is_win = sys.platform.startswith("win")
    is_mac = sys.platform == "darwin"
    is_arm = platform.machine() in ("arm64", "aarch64")

    if is_win:
        march = []
        mfma = []
        fastmath = []
        openmp_compile = ["/openmp"]
        openmp_link = []
        openmp_includes = []
        openmp_libraries = []
        lto = []
    else:
        # gcc / clang on Linux + macOS.
        if is_mac and is_arm:
            # Apple Silicon: clang doesn't support -march=native well; use
            # -mcpu=apple-m1 instead. AVX flags are x86-only, so skip them.
            march = ["-mcpu=apple-m1"]
            mfma = []
        else:
            march = ["-march=native"]
            mfma = ["-mfma", "-mprefer-vector-width=256"] if not is_arm else []

        fastmath = ["-ffast-math", "-fno-finite-math-only"]
        lto = ["-flto"]

        if is_mac:
            # Locate libomp. Try brew first, then homebrew defaults.
            libomp_prefix = None
            brew = shutil.which("brew")
            if brew:
                try:
                    libomp_prefix = subprocess.check_output(
                        [brew, "--prefix", "libomp"],
                        stderr=subprocess.DEVNULL,
                    ).decode().strip()
                except subprocess.CalledProcessError:
                    libomp_prefix = None
            if not libomp_prefix or not os.path.isdir(libomp_prefix):
                for candidate in ("/opt/homebrew/opt/libomp",
                                  "/usr/local/opt/libomp"):
                    if os.path.isdir(candidate):
                        libomp_prefix = candidate
                        break
            if libomp_prefix and os.path.isdir(libomp_prefix):
                openmp_compile = ["-Xpreprocessor", "-fopenmp"]
                openmp_link = ["-lomp"]
                openmp_includes = [os.path.join(libomp_prefix, "include")]
                openmp_libraries = [os.path.join(libomp_prefix, "lib")]
            else:
                # libomp missing: build without OpenMP. Single-threaded.
                openmp_compile = []
                openmp_link = []
                openmp_includes = []
                openmp_libraries = []
                print("setup.py: libomp not found on macOS; building without "
                      "OpenMP. `brew install libomp` to enable parallel fit.",
                      file=sys.stderr)
        else:
            # Linux.
            openmp_compile = ["-fopenmp"]
            openmp_link = ["-fopenmp"]
            openmp_includes = []
            openmp_libraries = []

    return dict(
        march=march, mfma=mfma, fastmath=fastmath, lto=lto,
        openmp_compile=openmp_compile, openmp_link=openmp_link,
        openmp_includes=openmp_includes, openmp_libraries=openmp_libraries,
    )


_PF = _platform_flags()

compile_args = (["-O3"] + _PF["march"] + _PF["mfma"] + _PF["fastmath"]
                + ["-funroll-loops", "-ftree-vectorize"]
                + _PF["lto"] + _PF["openmp_compile"])
link_args = _PF["lto"] + _PF["openmp_link"]


def make_ext(module_name, source_path, extra_sources=None):
    return Extension(
        name=module_name,
        sources=[source_path] + (extra_sources or []),
        language="c++",
        include_dirs=[np.get_include(), "mttrees/cfuncs_fast"]
                     + _PF["openmp_includes"],
        library_dirs=_PF["openmp_libraries"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )


# Slim release: the legacy fp64 Cython kernel under mttrees/cfuncs/ was
# dropped. Production uses cfuncs_cpp (selfhash); cfuncs_fast remains as the
# prebin fallback under MT_BUILD_FAST=1. Re-add a `cfuncs/` block here if a
# future revision needs the fp64 reference build.
extensions = []

# cfuncs_fast/: new backend under construction. Opt-in via MT_BUILD_FAST=1
# (default on) so collaborators can disable if build deps are missing. The
# Python dispatch in tree.py respects MT_BACKEND=legacy|fast at import time.
import os as _os
if _os.environ.get("MT_BUILD_FAST", "1") != "0":
    _simd_src = ["mttrees/cfuncs_fast/simd_helpers.cpp"]
    _cpp_src = ["mttrees/cfuncs_fast/sort_cpp.cpp"]
    extensions += [
        make_ext("mttrees.cfuncs_fast.mt_ctree_selfhash",
                 "mttrees/cfuncs_fast/mt_ctree_selfhash.pyx"),
        make_ext("mttrees.cfuncs_fast.mt_ctree_prebin",
                 "mttrees/cfuncs_fast/mt_ctree_prebin.pyx"),
        make_ext("mttrees.cfuncs_fast.mt_ctree_selfhash_f32",
                 "mttrees/cfuncs_fast/mt_ctree_selfhash_f32.pyx",
                 extra_sources=_simd_src + _cpp_src),
        make_ext("mttrees.cfuncs_fast.mt_ctree_prebin_f32",
                 "mttrees/cfuncs_fast/mt_ctree_prebin_f32.pyx",
                 extra_sources=_simd_src),
    ]

# Path B (C++ via pybind11). Gated by MT_BUILD_CPP=1 (default on if pybind11
# is available). The MT_BACKEND env var in tree.py dispatches between
# cfuncs (legacy), cfuncs_fast (cython M0-M5+M4), and cfuncs_cpp (this).
_cpp_extensions = []
if _HAVE_PYBIND11 and _os.environ.get("MT_BUILD_CPP", "1") != "0":
    _cpp_compile = (["-O3"] + _PF["march"] + _PF["mfma"] + _PF["fastmath"]
                    + ["-funroll-loops", "-ftree-vectorize"]
                    + _PF["openmp_compile"])
    _cpp_extensions = [
        Pybind11Extension(
            "mttrees.cfuncs_cpp.mt_ctree_selfhash_f32",
            sources=[
                "mttrees/cfuncs_cpp/mt_ctree_selfhash_f32.cpp",
                "mttrees/cfuncs_cpp/kernel/quantile_bins.cpp",
                "mttrees/cfuncs_cpp/kernel/fit.cpp",
            ],
            include_dirs=["mttrees/cfuncs_cpp"] + _PF["openmp_includes"],
            library_dirs=_PF["openmp_libraries"],
            extra_compile_args=_cpp_compile,
            extra_link_args=_PF["openmp_link"],
            cxx_std=17,
        ),
    ]

setup(
    name="mttrees",
    packages=[
        "mttrees",
        "mttrees.cfuncs_fast",
        "mttrees.cfuncs_cpp",
    ],
    ext_modules=(cythonize(extensions, language_level="3") if extensions else [])
                + _cpp_extensions,
)
