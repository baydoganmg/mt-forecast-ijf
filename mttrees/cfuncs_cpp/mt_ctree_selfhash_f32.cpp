// Path B Phase 1 skeleton: pybind11 module that exposes a stub
// fit_mt_tree_bin matching the cfuncs_fast signature.
//
// Future phases:
//   M2: port find_best_split_mod with std::stable_sort + OpenMP per-feature
//   M3: full fit_mt_tree_bin with quantile_bins (already in kernel/quantile_bins.hpp)
//   M4: predict_mt_tree
//   M5: verify O2/O5/O8 goldens pass; benchmark vs cfuncs_fast
//
// For now this module is a NOT_IMPLEMENTED-raising stub so the Python
// dispatcher can wire up MT_BACKEND=cpp and we can iterate.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "kernel/quantile_bins.hpp"
#include "kernel/fit.hpp"

namespace py = pybind11;

namespace mttrees::cpp_kernel {

// Test entry: stable quantile_bins exposed for unit testing the bin assignment
// against the cfuncs_fast Cython version. Same input -> same output is
// required before we trust the kernel layer.
py::tuple quantile_bins_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> arr,
    int n_bins
) {
    const int n_sample = static_cast<int>(arr.shape(0));
    auto bins        = py::array_t<int>(n_sample);
    auto bin_max     = py::array_t<float>(n_bins);
    auto bin_min     = py::array_t<float>(n_bins);
    auto bin_counts  = py::array_t<int>(n_bins);
    auto sort_idx    = py::array_t<int>(n_sample);
    std::vector<uint64_t> key_scratch(n_sample);

    QuantileBinsResult r = quantile_bins_f32(
        arr.data(), n_sample, n_bins,
        bins.mutable_data(),
        bin_max.mutable_data(),
        bin_min.mutable_data(),
        bin_counts.mutable_data(),
        sort_idx.mutable_data(),
        key_scratch.data()
    );
    return py::make_tuple(bins, bin_max, bin_min, bin_counts,
                          r.is_unique, r.resulting_bin_count);
}

py::tuple fit_mt_tree_bin_stub(py::args, py::kwargs) {
    throw std::runtime_error(
        "mttrees.cfuncs_cpp.mt_ctree_selfhash_f32.fit_mt_tree_bin: "
        "not yet implemented. Path B is in development; use "
        "MT_BACKEND=fast (cfuncs_fast) for now.");
}

}  // namespace mttrees::cpp_kernel

PYBIND11_MODULE(mt_ctree_selfhash_f32, m) {
    m.doc() = "Path B C++ selfhash kernel (skeleton). See cfuncs_fast for the "
              "production backend until M5 of Path B lands.";

    m.def("quantile_bins", &mttrees::cpp_kernel::quantile_bins_py,
          py::arg("arr"), py::arg("n_bins"),
          "Quantile binning over a 1D fp32 array. Returns "
          "(bins, bin_max, bin_min, bin_counts, is_unique, resulting_bin_count). "
          "Bit-equivalent contract to cfuncs_fast _quantile_bins_nogil.");

    m.def("predict_mt_tree", &mttrees::cpp_kernel::predict_mt_tree,
          py::arg("features"), py::arg("node_stats"),
          py::arg("split_vals"), py::arg("node_means"),
          py::arg("maxdepth"), py::arg("verbose"), py::arg("num_threads"),
          "Path B predict: traverses each row through the fitted tree.");

    m.def("fit_mt_tree_bin", &mttrees::cpp_kernel::fit_mt_tree_bin,
          py::arg("features"), py::arg("labels"),
          py::arg("weights_target"), py::arg("target_weight_map"),
          py::arg("n_objective"), py::arg("objective_weights"),
          py::arg("max_depth"), py::arg("mtry"),
          py::arg("samp_feat_by_node"), py::arg("min_data_in_leaf"),
          py::arg("num_nodes"), py::arg("min_gain_to_split"),
          py::arg("num_bins"), py::arg("num_threads"),
          py::arg("verbose"), py::arg("agg_mode"),
          py::arg("obj_weight_mode") = 0,
          py::arg("obj_weight_dirichlet_alpha") = 1.0f,
          py::arg("rng") = py::none(),
          py::arg("cxx_rng_seed") = -1,
          "Path B selfhash fit; matches cfuncs_fast contract at nt=1. "
          "obj_weight_mode: 0=FIXED (default, bit-exact to prior contract), "
          "1=DIRICHLET (per-node resample), 2=ONEHOT (per-node uniform pick). "
          "obj_weight_dirichlet_alpha: concentration parameter for mode=1.");

    m.def("ws_capacity_bytes", &mttrees::cpp_kernel::ws_capacity_bytes,
          "Test-only: bytes held by the calling thread's per-feature scan "
          "workspace. Tests use this to verify the adaptive shrink at the "
          "end of fit_mt_tree_bin.");

    m.def("fit_mt_tree_bin_indirect",
          &mttrees::cpp_kernel::fit_mt_tree_bin_indirect,
          py::arg("features_base"), py::arg("labels_base"),
          py::arg("positions"),
          py::arg("weights_target"), py::arg("target_weight_map"),
          py::arg("n_objective"), py::arg("objective_weights"),
          py::arg("max_depth"), py::arg("mtry"),
          py::arg("samp_feat_by_node"), py::arg("min_data_in_leaf"),
          py::arg("num_nodes"), py::arg("min_gain_to_split"),
          py::arg("num_bins"), py::arg("num_threads"),
          py::arg("verbose"), py::arg("agg_mode"),
          py::arg("obj_weight_mode") = 0,
          py::arg("obj_weight_dirichlet_alpha") = 1.0f,
          py::arg("rng") = py::none(),
          py::arg("cxx_rng_seed") = -1,
          "Indirected fit: same tree as fit_mt_tree_bin applied to "
          "features_base[positions], labels_base[positions]. Saves per-tree "
          "worker materialization (design note). obj_weight_mode / "
          "obj_weight_dirichlet_alpha: see fit_mt_tree_bin docs.");
}
