#pragma once

#include <cstdint>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace mttrees::cpp_kernel {

/* Path B fit_mt_tree_bin: pure-C++ implementation of the selfhash tree fit.
 *
 * Matches the cfuncs_fast.mt_ctree_selfhash_f32.fit_mt_tree_bin signature
 * exactly (so the Python ensemble layer can dispatch via MT_BACKEND=cpp
 * without other changes).
 *
 * Returns a tuple (tree_info, split_vals, node_means) where:
 *   tree_info shape [actual_nodes, 7], int32
 *   split_vals shape [actual_nodes], float32
 *   node_means shape [actual_nodes, K], float32
 *
 * tree_info column layout (matches cfuncs_fast / cfuncs legacy):
 *   col 0: node_id
 *   col 1: n_samples in node
 *   col 2: node_status (-1=terminal, -2=tosplit, -3=interior)
 *   col 3: depth
 *   col 4: split_var
 *   col 5: left_child_id
 *   col 6: right_child_id
 */
/* Path B predict_mt_tree: traverse the fitted tree for each row of features,
 * return the leaf's node_means as the per-row K-vector prediction.
 *
 * Matches the cfuncs_fast predict_mt_tree signature exactly:
 *   (features [N, P], node_stats [num_nodes, 7], split_vals [num_nodes],
 *    node_means [num_nodes, K], maxdepth, verbose, num_threads)
 *
 * Returns float32[N, K] predictions.
 *
 * NOTE: the `num_threads` argument is accepted for ABI parity with the
 * cfuncs_fast (Cython) predict_mt_tree call site in mttrees/tree.py
 * (the mmap-design review).
 * The cpp predict path is currently serial; the argument is ignored. Do
 * not remove it without also updating the dispatcher in tree.py.
 */
py::array_t<float> predict_mt_tree(
    py::array_t<const float, py::array::c_style | py::array::forcecast> features,
    py::array_t<const int, py::array::c_style | py::array::forcecast> node_stats,
    py::array_t<const float, py::array::c_style | py::array::forcecast> split_vals,
    py::array_t<const float, py::array::c_style | py::array::forcecast> node_means,
    int maxdepth,
    int verbose,
    int num_threads
);

py::tuple fit_mt_tree_bin(
    py::array_t<float, py::array::c_style | py::array::forcecast> features,
    py::array_t<float, py::array::c_style | py::array::forcecast> labels,
    py::array_t<float> weights_target,
    py::array_t<int> target_weight_map,
    int n_objective,
    py::array_t<float> objective_weights,
    int max_depth,
    float mtry,
    int samp_feat_by_node,
    int min_data_in_leaf,
    int num_nodes,
    float min_gain_to_split,
    int num_bins,
    int num_threads,
    int verbose,
    int agg_mode,
    int obj_weight_mode,
    float obj_weight_dirichlet_alpha,
    py::object rng = py::none(),
    int64_t cxx_rng_seed = -1
);

/* fit_mt_tree_bin_indirect: indirected-access variant of fit_mt_tree_bin.
 *
 * Takes the FULL base features/labels (typically a worker-side memmap view of
 * the shared training data) plus an `int32` positions array specifying which
 * rows belong to this tree's bagging sample. Saves ~80% × N × P × 4 bytes per
 * worker by avoiding the pandas `.loc[sample_index].to_numpy()` materialization
 * that the contiguous entry requires (design review
 * review_mmap_design_input_2026_06_04.md, item #3).
 *
 * Invariant (gated by tests/test_cpp_fit_indirect.py): for any positions[],
 *   fit_mt_tree_bin_indirect(x_base, y_base, positions, **kw)
 *   == fit_mt_tree_bin(x_base[positions], y_base[positions], **kw)
 * bit-exact (same float arithmetic on the same float values, in the same
 * order — the kernel iterates positions in input order).
 */
py::tuple fit_mt_tree_bin_indirect(
    py::array_t<const float, py::array::c_style | py::array::forcecast> features_base,
    py::array_t<const float, py::array::c_style | py::array::forcecast> labels_base,
    py::array_t<const int, py::array::c_style | py::array::forcecast> positions,
    py::array_t<float> weights_target,
    py::array_t<int> target_weight_map,
    int n_objective,
    py::array_t<float> objective_weights,
    int max_depth,
    float mtry,
    int samp_feat_by_node,
    int min_data_in_leaf,
    int num_nodes,
    float min_gain_to_split,
    int num_bins,
    int num_threads,
    int verbose,
    int agg_mode,
    int obj_weight_mode,
    float obj_weight_dirichlet_alpha,
    py::object rng = py::none(),
    int64_t cxx_rng_seed = -1
);

// Test-only introspection: bytes held by the calling thread's per-feature
// scan workspace (the static thread-local ScanWorkspace in fit.cpp).
// Used by tests/test_phase2.py to verify the adaptive shrink at the end of
// fit_mt_tree_bin actually frees memory when a large fit is followed by a
// smaller one.
size_t ws_capacity_bytes();

}  // namespace mttrees::cpp_kernel
