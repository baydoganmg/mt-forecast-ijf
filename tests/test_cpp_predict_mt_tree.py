"""TDD test suite for Path B predict_mt_tree.

Contract: bit-identical predictions vs cfuncs_fast on a tree fitted by EITHER
backend (since fit is bit-equivalent at nt=1).

Strategy:
- Fit a small tree using cfuncs_fast (the reference).
- Feed the same features to predict on both backends, compare bit-identically.

Run:
    python tests/test_cpp_predict_mt_tree.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np


def _fit_reference_tree(features, labels, max_depth=3, num_bins=4,
                        n_objective=1, agg_mode=0):
    from mttrees.cfuncs_fast.mt_ctree_selfhash_f32 import fit_mt_tree_bin
    K = labels.shape[1]
    weights_target = np.ones(K, dtype=np.float32)
    target_weight_map = np.zeros(K, dtype=np.intc)
    objective_weights = np.array([1.0] * max(1, n_objective), dtype=np.float32)
    np.random.seed(0)
    return fit_mt_tree_bin(
        features, labels, weights_target, target_weight_map, n_objective,
        objective_weights, max_depth, 1.0, 0, 5, 7, 0.0, num_bins, 1, 0, agg_mode,
    )


def _predict_cy(features, tree_info, split_vals, node_means, depth):
    from mttrees.cfuncs_fast.mt_ctree_selfhash_f32 import predict_mt_tree
    feat_np = np.ascontiguousarray(features, dtype=np.float32)
    return np.asarray(predict_mt_tree(
        feat_np, tree_info, split_vals, node_means, depth, 0)).copy()


def _predict_cpp(features, tree_info, split_vals, node_means, depth):
    from mttrees.cfuncs_cpp.mt_ctree_selfhash_f32 import predict_mt_tree
    feat_np = np.ascontiguousarray(features, dtype=np.float32)
    return np.asarray(predict_mt_tree(
        feat_np, tree_info, split_vals, node_means, depth, 0, 1)).copy()


def _compare_one(label, features, labels, max_depth, num_bins, n_obj, agg_mode,
                 X_test):
    tree_info, split_vals, node_means = _fit_reference_tree(
        features, labels, max_depth=max_depth, num_bins=num_bins,
        n_objective=n_obj, agg_mode=agg_mode)
    tree_info = np.ascontiguousarray(tree_info, dtype=np.intc)
    split_vals = np.ascontiguousarray(split_vals, dtype=np.float32)
    node_means = np.ascontiguousarray(node_means, dtype=np.float32)
    depth = max_depth
    cy_pred = _predict_cy(X_test, tree_info, split_vals, node_means, depth)
    try:
        cpp_pred = _predict_cpp(X_test, tree_info, split_vals, node_means, depth)
    except RuntimeError as exc:
        return False, f"cpp side raised: {exc}"
    except AttributeError as exc:
        return False, f"cpp side missing predict: {exc}"
    if cpp_pred.shape != cy_pred.shape:
        return False, f"shape cpp={cpp_pred.shape} cy={cy_pred.shape}"
    if not np.array_equal(cpp_pred, cy_pred):
        diff = np.abs(cpp_pred - cy_pred).max()
        return False, f"predictions diverge: Linf={diff:.3e}"
    return True, None


def main():
    rng = np.random.default_rng(42)
    cases = [
        ("n=40 P=3 K=4 depth=3 n_obj=1",
         dict(features=rng.standard_normal((40, 3)).astype(np.float32),
              labels=rng.standard_normal((40, 4)).astype(np.float32),
              max_depth=3, num_bins=4, n_obj=1, agg_mode=0)),
        ("n=100 P=5 K=6 depth=4 n_obj=1",
         dict(features=rng.standard_normal((100, 5)).astype(np.float32),
              labels=rng.standard_normal((100, 6)).astype(np.float32),
              max_depth=4, num_bins=8, n_obj=1, agg_mode=0)),
        ("n=80 P=4 K=6 multi-obj rank",
         dict(features=rng.standard_normal((80, 4)).astype(np.float32),
              labels=rng.standard_normal((80, 6)).astype(np.float32),
              max_depth=4, num_bins=8, n_obj=3, agg_mode=0)),
    ]
    rng_test = np.random.default_rng(99)
    failures = []
    for label, kw in cases:
        features = np.ascontiguousarray(kw.pop("features"))
        labels = np.ascontiguousarray(kw.pop("labels"))
        X_test = rng_test.standard_normal((20, features.shape[1])).astype(np.float32)
        ok, msg = _compare_one(label, features, labels,
                                kw["max_depth"], kw["num_bins"],
                                kw["n_obj"], kw["agg_mode"], X_test)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label}")
        if not ok:
            print(f"       {msg}")
            failures.append(label)
    if failures:
        print(f"\n{len(failures)} failing case(s).")
        sys.exit(1)
    print("\nAll predict_mt_tree TDD cases pass.")


if __name__ == "__main__":
    main()
