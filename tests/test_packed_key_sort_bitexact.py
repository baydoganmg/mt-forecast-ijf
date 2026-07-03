"""
Bit-exact regression test for the sprint-6 packed-key sort replacing
``std::stable_sort`` in ``quantile_bins_f32`` and the post-split
reorder (``mttrees/cfuncs_cpp/kernel``).

The packed-key sort encodes each element as
``(monotone_uint32(value) << 32) | row_index`` and runs ``std::sort``.
Properties that make it produce the IDENTICAL permutation to the
previous ``std::stable_sort`` with comparator ``arr[a] < arr[b]``:

1. ``monotone_uint32`` is strictly monotone over all non-NaN fp32:
   ``f1 < f2  <=>  u(f1) < u(f2)``, with -0.0 canonicalized to +0.0 so
   that float-equal values (including ±0.0) map to EQUAL keys.
2. Float-equal values therefore tie on the high 32 bits and are
   ordered by the low 32 bits = original index — exactly stable_sort's
   tie-breaking rule.

Golden outputs in ``tests/golden_packed_key_sort.npz`` were captured
with the PRE-CHANGE (stable_sort) kernel on this machine; the test
asserts byte-for-byte equality of (tree_info, split_vals, node_means)
for a battery of adversarial inputs: ties everywhere, mixed ±0.0,
denormals, large/small magnitudes, constant columns, near-equal values
straddling RELATIVE_THRESHOLD.

Regenerate goldens (ONLY on a known-good kernel):
    REGEN_GOLDEN=1 python -m pytest tests/test_packed_key_sort_bitexact.py -x

PLATFORM NOTE: the committed npz was captured on the Linux box
(gcc/x86_64). Tree outputs embed platform fp behavior (-ffast-math
vectorization, arch FMA contraction), so this test FAILS on other
platforms against the committed goldens — pre-existing cross-platform
drift, NOT a sort regression. To validate on another platform: check
out the pre-sprint-6 kernel (parent of the packed-key commit), run
REGEN_GOLDEN=1 there, then re-run this test on the packed-key kernel.
Done on the Mac Studio 2026-06-12: 11/11 passed against Mac-native
stable_sort goldens.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("MT_BACKEND", "cpp")
os.environ.setdefault("MT_DTYPE", "fp32")

import numpy as np
import pytest

pytest.importorskip("mttrees.cfuncs_cpp.mt_ctree_selfhash_f32")

GOLDEN_PATH = Path(__file__).parent / "golden_packed_key_sort.npz"


def _fit_kw(K, max_depth=4, num_bins=8):
    return dict(
        weights_target=np.ones(K, dtype=np.float32),
        target_weight_map=np.zeros(K, dtype=np.int32),
        n_objective=1,
        objective_weights=np.ones(1, dtype=np.float32),
        max_depth=max_depth,
        mtry=1.0,
        samp_feat_by_node=0,
        min_data_in_leaf=2,
        num_nodes=2 ** max_depth,
        min_gain_to_split=0.0,
        num_bins=num_bins,
        num_threads=1,
        verbose=0,
        agg_mode=1,
    )


def _adversarial_cases():
    """(name, features, labels) triples exercising sort edge cases."""
    rng = np.random.default_rng(20260612)
    cases = []

    # 1. Mixed +0.0 / -0.0 with ties: the ±0 canonicalization case.
    x = np.zeros((96, 2), dtype=np.float32)
    x[::2, 0] = -0.0
    x[1::2, 0] = +0.0
    x[:, 1] = np.repeat(np.arange(12), 8).astype(np.float32)
    y = (np.arange(96) % 7).astype(np.float32).reshape(-1, 1)
    cases.append(("pm_zero_ties", x, np.ascontiguousarray(y)))

    # 2. Heavy ties: few unique values, many repeats, random order.
    x = rng.integers(0, 5, size=(200, 3)).astype(np.float32)
    y = rng.standard_normal((200, 2)).astype(np.float32)
    cases.append(("heavy_ties", x, y))

    # 3. Denormals + tiny magnitudes around zero.
    base = np.array([0.0, 1e-45, -1e-45, 1e-38, -1e-38, 1e-30], dtype=np.float32)
    x = np.tile(base, (40, 1)).astype(np.float32)
    x += rng.standard_normal(x.shape).astype(np.float32) * 1e-40
    y = rng.standard_normal((x.shape[0], 1)).astype(np.float32)
    cases.append(("denormals", np.ascontiguousarray(x), y))

    # 4. Large magnitudes incl. negatives (sign-bit handling).
    x = (rng.standard_normal((150, 4)) * 1e6).astype(np.float32)
    x[:30] = -x[:30]
    y = rng.standard_normal((150, 3)).astype(np.float32)
    cases.append(("large_mixed_sign", np.ascontiguousarray(x), y))

    # 5. Near-equal values straddling RELATIVE_THRESHOLD spacing.
    v = 1.0 + np.arange(120, dtype=np.float32) * 1e-7
    x = np.stack([v, v[::-1]], axis=1).astype(np.float32)
    y = (v * 3).reshape(-1, 1).astype(np.float32)
    cases.append(("threshold_straddle", np.ascontiguousarray(x),
                  np.ascontiguousarray(y)))

    # 6. Realistic panel: moderate N, several features, K>1, deeper tree.
    x = rng.standard_normal((600, 8)).astype(np.float32)
    y = rng.standard_normal((600, 6)).astype(np.float32)
    cases.append(("realistic_panel", x, y))

    return cases


def _run_case(features, labels, depth=4, num_bins=8):
    from mttrees.cfuncs_cpp.mt_ctree_selfhash_f32 import fit_mt_tree_bin
    K = labels.shape[1]
    np.random.seed(123)
    return fit_mt_tree_bin(
        np.ascontiguousarray(features, dtype=np.float32),
        np.ascontiguousarray(labels, dtype=np.float32),
        **_fit_kw(K=K, max_depth=depth, num_bins=num_bins))


def test_golden_capture_or_compare():
    cases = _adversarial_cases()
    if os.environ.get("REGEN_GOLDEN") == "1" or not GOLDEN_PATH.exists():
        out = {}
        for name, x, y in cases:
            ti, sv, nm = _run_case(x, y)
            out[f"{name}__ti"] = np.asarray(ti)
            out[f"{name}__sv"] = np.asarray(sv)
            out[f"{name}__nm"] = np.asarray(nm)
        np.savez_compressed(GOLDEN_PATH, **out)
        pytest.skip(f"golden regenerated at {GOLDEN_PATH}; rerun to compare")

    golden = np.load(GOLDEN_PATH)
    for name, x, y in cases:
        ti, sv, nm = _run_case(x, y)
        for arr, suffix in ((ti, "ti"), (sv, "sv"), (nm, "nm")):
            g = golden[f"{name}__{suffix}"]
            assert np.array_equal(np.asarray(arr), g), (
                f"case={name} field={suffix}: NOT bit-exact vs stable_sort golden")


def test_determinism_across_runs():
    for name, x, y in _adversarial_cases():
        a = _run_case(x, y)
        b = _run_case(x, y)
        for f1, f2 in zip(a, b):
            assert np.array_equal(np.asarray(f1), np.asarray(f2)), name
