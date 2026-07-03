#include "fit.hpp"
#include "quantile_bins.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include <omp.h>

namespace mttrees::cpp_kernel {

namespace {

constexpr int NODE_TERMINAL = -1;
constexpr int NODE_TOSPLIT = -2;
constexpr int NODE_INTERIOR = -3;

// MT_CPP_BIN_MODE selects between two correctness-equivalent bin-policy
// modes for the selfhash kernel. Both modes use the same right-child
// guard in the scoring loop; they differ only in the bin count used for
// the per-feature quantile partition:
//
//   adaptive   (default) -> B_node = min(num_bins, n_sample/min_data_in_leaf)
//                           coarsens the candidate grid in small/deep nodes
//   fixed_valid          -> B_node = num_bins; rely entirely on the
//                           right-child guard during scoring to drop
//                           invalid candidates
//
// The fixed_valid mode exists for the audit comparator: it isolates the
// "leaf-size invariant fix" from the "adaptive grid coarsening." It is
// not a production knob.
enum class BinMode { Adaptive, FixedValid };

inline BinMode read_bin_mode() {
    const char * env = std::getenv("MT_CPP_BIN_MODE");
    if (env != nullptr && std::strcmp(env, "fixed_valid") == 0) {
        return BinMode::FixedValid;
    }
    return BinMode::Adaptive;
}

// M4 port: thread-local scratch for the per-feature scan. Each OpenMP
// thread keeps its own persistent vectors that are resize()'d (no-op when
// already the right size) at the top of each feature's scoring block.
// Allocation amortized across all nodes / trees / fits on the same thread.
struct ScanWorkspace {
    std::vector<float> temp_x;
    std::vector<int> bin_rep;
    std::vector<int> bin_instance_count;
    std::vector<float> bin_max;
    std::vector<float> bin_min;
    std::vector<float> total_sum;
    std::vector<float> left_sum;
    std::vector<int> qb_sort_idx;
    // sprint-6: packed-key scratch for the quantile_bins argsort. One
    // per-thread buffer, overwritten per feature (the cache-friendly
    // reuse pattern — NOT a per-feature slab; see the sprint-3 revert).
    std::vector<uint64_t> qb_keys;

    // Total capacity in bytes — for the ws_capacity_bytes() introspection
    // helper used by tests/test_phase2.py to verify adaptive shrinkage at
    // dataset boundaries.
    size_t capacity_bytes() const {
        return temp_x.capacity() * sizeof(float)
             + bin_rep.capacity() * sizeof(int)
             + bin_instance_count.capacity() * sizeof(int)
             + bin_max.capacity() * sizeof(float)
             + bin_min.capacity() * sizeof(float)
             + total_sum.capacity() * sizeof(float)
             + left_sum.capacity() * sizeof(float)
             + qb_sort_idx.capacity() * sizeof(int)
             + qb_keys.capacity() * sizeof(uint64_t);
    }
    // Adaptive shrink. Called at the end of fit_mt_tree_bin when the
    // workspace held over from a previous fit may be much larger than the
    // current fit needs. Skipped (O(1) check) when capacity is already
    // close to the current root size, so hot-path cost is negligible.
    void shrink_if_oversized(int n_sample_root) {
        const size_t hi = std::max<size_t>(
            16, 2 * (size_t)std::max(1, n_sample_root));
        if (temp_x.capacity() > hi) {
            std::vector<float>(temp_x).swap(temp_x);
            std::vector<int>(bin_rep).swap(bin_rep);
            std::vector<int>(qb_sort_idx).swap(qb_sort_idx);
            std::vector<uint64_t>(qb_keys).swap(qb_keys);
            // bin_* / total_sum / left_sum are O(num_bins) or O(K), tiny;
            // skip those to keep this call cheap.
        }
    }
};

// Namespace-scope per-thread workspace. Same lifetime as the previous
// function-local static thread_local declaration, but reachable from the
// shrink check at the end of fit_mt_tree_bin and from ws_capacity_bytes().
thread_local ScanWorkspace tw_global;

// Drop-in for the cfuncs_fast sample_features helper. Uses the global numpy
// RNG via Python interop so the M0 RNG contract carries over: caller does
// np.random.seed(per_tree_seed) before fit, and this picks up the same draws.
//
// Called once PER NODE when samp_feat_by_node==1 (the RF path), i.e.
// ~nodes x trees times per forest -- a hot Python-interop site that mDT/mGBT
// (sample-once-at-root) never hit. Two perf changes, both bit-exact:
//  (1) Use numpy.random.permutation(P)[:n_features] instead of
//      numpy.random.choice(P, size=n_features, replace=False). numpy's
//      legacy RandomState.choice(replace=False, p=None) is IMPLEMENTED as
//      permutation(pop_size)[:size], so this draws the identical indices AND
//      leaves the global RNG in the identical state (verified empirically:
//      same result + same state for every (P, n_features) that arises), while
//      skipping choice's per-call validation overhead (~4x cheaper per call).
//  (2) Cache the bound permutation callable once. It always operates on
//      numpy's global RandomState singleton (the one np.random.seed mutates),
//      so caching the function reference does not change which state each call
//      reads -- still bit-exact.
void sample_features_np(const py::object & perm, int P, int n_features, int * out) {
    // `perm`: a per-tree-RNG bound .permutation callable (RandomState.permutation),
    // or None to fall back to numpy's GLOBAL RNG (legacy / random_state=None path).
    // The per-tree callable is what makes the threading backend race-free AND
    // bit-exact: RandomState(s).permutation == np.random.seed(s)+np.random.permutation.
    py::object result;
    if (perm.is_none()) {
        // Heap-allocated (intentionally leaked) so the cached callable's
        // destructor never runs. A plain `static py::object` would Py_DECREF
        // during C++ static teardown -- AFTER Py_Finalize -- which aborts the
        // process at exit. Leaking one object for the process lifetime is the
        // standard pybind11 idiom for interpreter-lifetime caches.
        static const py::object* permutation = new py::object(
            py::module_::import("numpy").attr("random").attr("permutation"));
        result = (*permutation)(P);
    } else {
        result = perm(P);
    }
    auto arr = result.cast<py::array_t<int64_t>>();
    const int64_t * data = static_cast<const int64_t *>(arr.request().ptr);
    for (int i = 0; i < n_features; ++i) {
        out[i] = static_cast<int>(data[i]);
    }
}

// Pure-C++ feature subsample: pick `n_features` of [0,P) WITHOUT replacement
// via a partial Fisher-Yates from a per-tree std::mt19937. Touches NO Python,
// so the whole fit can run GIL-free -> the threading backend parallelises at
// full speed (no per-node numpy callback to serialise on the GIL). NOT
// bit-exact to numpy's permutation (different RNG) -- intended only for the RF
// threading path, where a different valid random subsample yields a
// statistically-equivalent ensemble. mt19937 + manual modulo are both
// platform-independent, so the threaded forest is deterministic AND identical
// on Linux/Mac (modulo bias is negligible for the small P that arises and is
// irrelevant once bit-exactness is dropped). `pool` is caller scratch (len P).
void sample_features_cxx(std::mt19937 & gen, std::vector<int> & pool,
                         int P, int n_features, int * out) {
    for (int i = 0; i < P; ++i) pool[i] = i;
    for (int i = 0; i < n_features; ++i) {
        const uint32_t r = static_cast<uint32_t>(gen());
        const int j = i + static_cast<int>(r % static_cast<uint32_t>(P - i));
        std::swap(pool[i], pool[j]);
        out[i] = pool[i];
    }
}

// Per-node objective-weight resampling helpers. Both use the global numpy RNG
// (same M0 contract as sample_features_np) so that per-tree seeding via
// np.random.seed(per_tree_seed) drives the per-node draws deterministically.
//
// sample_dirichlet_via_np: draws w ~ Dirichlet(alpha * ones(n_objective)) and
// writes the result into out (length n_objective).
void sample_dirichlet_via_np(const py::object & rng, int n_objective, float alpha, float * out) {
    auto np = py::module_::import("numpy");
    py::object dirichlet = rng.is_none()
        ? np.attr("random").attr("dirichlet") : rng.attr("dirichlet");
    // alpha vector: np.ones(n_objective) * alpha
    py::object ones = np.attr("ones");
    py::object alpha_vec = ones(n_objective).attr("__mul__")(alpha);
    py::object result = dirichlet(alpha_vec);
    auto arr = result.cast<py::array_t<double>>();
    for (int i = 0; i < n_objective; ++i) {
        out[i] = static_cast<float>(arr.at(i));
    }
}

// sample_categorical_via_np: returns a single integer drawn uniformly from
// {0, ..., n_objective - 1} via np.random.choice(n_objective).
int sample_categorical_via_np(const py::object & rng, int n_objective) {
    auto np = py::module_::import("numpy");
    py::object choice = rng.is_none()
        ? np.attr("random").attr("choice") : rng.attr("choice");
    py::object result = choice(n_objective);
    // np.random.choice with an integer argument returns a 0-d int array.
    return static_cast<int>(result.cast<int64_t>());
}

// find_best_split_mod inlined: scans all sampled features at the current
// node, fills split_info / split_perf / aggregated_objectives, applies the
// rank or score aggregation, picks the best (split_var, split_val),
// partitions row_indices for the children. Returns (last_index, split_var,
// split_val); split_var == -1 means no valid split (terminal).
struct BestSplit {
    int last_index;
    int split_var;
    float split_val;
};

BestSplit find_best_split_mod(
    const float * features, int N, int P,
    const float * labels, int K,
    int * row_indices,
    int n_sample, int n_predictor, int n_objective,
    int ndstart, int ndend,
    const int * feature_indices,
    int min_data_in_leaf,
    const float * weights_target,
    const int * target_weight_map,
    const float * objective_weights,
    int num_bins,
    int agg_mode,
    int num_threads,
    float min_gain_to_split,
    // workspace
    std::vector<float> & critParent,
    std::vector<float> & overall_total_sum,
    std::vector<float> & aggregated_objectives,  // (n_predictor*num_bins, n_objective)
    std::vector<float> & objective_weighted_rank,
    std::vector<int> & sort_idx_obj,
    std::vector<int> & rank_obj,
    std::vector<int> & split_info_var,      // (n_predictor * num_bins), holds feat_idx or -1
    std::vector<int> & split_info_bin_count_left,  // (n_predictor * num_bins)
    std::vector<int> & split_info_bin_count_right,
    std::vector<float> & split_info_split_val,
    std::vector<int> & sort_scratch
) {
    // Bin-policy mode (read at every call; env-var driven so tests can
    // swap without rebuild). The right-child guard below is shared.
    const BinMode bin_mode = read_bin_mode();
    int B_node;
    if (bin_mode == BinMode::Adaptive) {
        // Adaptive B per node: clamp the requested bin count to
        // B_node = min(num_bins, n_sample / min_data_in_leaf). Under
        // an even quantile partition this bounds the nominal candidate
        // grid so most candidates respect min_data_in_leaf on both
        // sides. The bound is NOT sufficient on its own: with duplicate
        // feature values quantile_bins collapses bins (one bin can hold
        // the vast majority of rows), and after that bin's contribution
        // to cumul_count the right child can still fall below
        // min_data_in_leaf. The right-child guard below remains part of
        // the correctness contract for that case.
        B_node = std::min(num_bins, n_sample / min_data_in_leaf);
    } else {
        // fixed_valid: keep the requested num_bins as-is. All
        // leaf-size enforcement comes from the right-child guard
        // during scoring. This is the minimal "bug-fix only" form of
        // the selfhash patch and serves as a comparator to isolate
        // the adaptive grid coarsening effect.
        B_node = num_bins;
    }
    // If we don't have enough samples for at least two valid bins on
    // either side, the node has no valid split; mark terminal.
    if (B_node < 2 || n_sample < 2 * min_data_in_leaf) {
        return {-1, -1, -1.0f};
    }

    // Parent loss (fp32 to match cfuncs_fast).
    // min-gain split pruning groundwork: only when enabled (>0) do we
    // accumulate per-target sum of squares, so the default path is
    // unchanged in both cost and result.
    std::vector<float> sumsq;
    if (min_gain_to_split > 0.0f) {
        sumsq.assign(K, 0.0f);
    }
    for (int t = 0; t < K; ++t) overall_total_sum[t] = 0.0f;
    for (int i = ndstart; i < ndend; ++i) {
        const int ri = row_indices[i];
        for (int t = 0; t < K; ++t) {
            overall_total_sum[t] += labels[ri * K + t];
            if (min_gain_to_split > 0.0f) {
                sumsq[t] += labels[ri * K + t] * labels[ri * K + t];
            }
        }
    }
    for (int t = 0; t < K; ++t) {
        critParent[t] = overall_total_sum[t] * overall_total_sum[t] / float(n_sample);
    }

    // Reset aggregated_objectives + objective_weighted_rank.
    const int n_cand = n_predictor * B_node;
    for (int i = 0; i < n_cand; ++i) {
        objective_weighted_rank[i] = 0.0f;
        for (int o = 0; o < n_objective; ++o) {
            aggregated_objectives[i * n_objective + o] = 0.0f;
        }
        split_info_var[i] = -1;
        split_info_bin_count_left[i] = 0;
        split_info_bin_count_right[i] = 0;
        split_info_split_val[i] = 0.0f;
    }

    int unique_counts = 0;

    // M4: per-feature OpenMP. Writes to aggregated_objectives /
    // split_info_* are partitioned by feature index p (each thread owns
    // indices [p*B_node, (p+1)*B_node)), so no synchronization is needed.
    // Per-feature scratch (temp_x, bin_rep, ..., left_sum) is held in a
    // thread_local ScanWorkspace so each thread keeps its own buffers
    // amortized across calls.
    #pragma omp parallel num_threads(num_threads) if (num_threads > 1) \
            reduction(+:unique_counts)
    {
        ScanWorkspace& tw = tw_global;
        tw.temp_x.resize(n_sample);
        tw.bin_rep.resize(n_sample);
        tw.bin_instance_count.resize(B_node);
        tw.bin_max.resize(B_node);
        tw.bin_min.resize(B_node);
        tw.total_sum.resize(B_node * K);
        tw.left_sum.resize(K);
        tw.qb_sort_idx.resize(n_sample);
        tw.qb_keys.resize(n_sample);

        #pragma omp for schedule(static)
        for (int p = 0; p < n_predictor; ++p) {
            const int feat_idx = feature_indices[p];

            for (int i = 0; i < n_sample; ++i) {
                tw.temp_x[i] = features[row_indices[ndstart + i] * P + feat_idx];
            }

            QuantileBinsResult qres = quantile_bins_f32(
                tw.temp_x.data(), n_sample, B_node,
                tw.bin_rep.data(),
                tw.bin_max.data(), tw.bin_min.data(),
                tw.bin_instance_count.data(),
                tw.qb_sort_idx.data(), tw.qb_keys.data());

            // Reset total_sum for this feature.
            for (int b = 0; b < B_node; ++b) {
                for (int t = 0; t < K; ++t) tw.total_sum[b * K + t] = 0.0f;
            }

            if (qres.is_unique == 1) {
                unique_counts += 1;
                continue;
            }

            // Accumulate total_sum[bin_rep[i], t] += labels[row_indices[ndstart+i], t]
            // sprint-4: K==1 (mGBT / single-objective) fast path drops the inner
            // `for t` loop. With K=1 the body is one add — generic loop costs a
            // cmp/jge/inc per outer iter even when K is loop-invariant, because
            // it's a runtime arg (no constant-folding across the pybind11 call).
            // Bit-exact by direct substitution: t=0, b*K+t=b, ri*K+t=ri.
            if (K == 1) {
                for (int i = 0; i < n_sample; ++i) {
                    const int ri = row_indices[ndstart + i];
                    const int b = tw.bin_rep[i];
                    tw.total_sum[b] += labels[ri];
                }
            } else {
                for (int i = 0; i < n_sample; ++i) {
                    const int ri = row_indices[ndstart + i];
                    const int b = tw.bin_rep[i];
                    for (int t = 0; t < K; ++t) {
                        tw.total_sum[b * K + t] += labels[ri * K + t];
                    }
                }
            }

            // Scan bins for split candidates.
            for (int t = 0; t < K; ++t) tw.left_sum[t] = 0.0f;
            int cumul_count = 0;
            for (int b = 0; b < qres.resulting_bin_count - 1; ++b) {
                cumul_count += tw.bin_instance_count[b];
                for (int t = 0; t < K; ++t) {
                    tw.left_sum[t] += tw.total_sum[b * K + t];
                }
                // Left-child contract: never emit a candidate whose left side
                // is below min_data_in_leaf. The prior inner condition
                // (`cumul_count + bin_instance_count[b+1] < n_sample`) only
                // skipped when the NEXT bin would still leave a non-empty
                // right side; when the next bin consumed everything it fell
                // through and emitted an underfilled left split. Exercised by
                // tests/test_phase2.py::test_selfhash_min_data_in_leaf_respected_on_tieheavy[fixed_valid].
                if (cumul_count < min_data_in_leaf) continue;
                // Right-child contract: adaptive B_node guarantees this under
                // BinMode=adaptive, but BinMode=fixed_valid (and tied-value
                // bin collapse) can still drop us below; explicit break keeps
                // the contract in every mode.
                if (n_sample - cumul_count < min_data_in_leaf) break;

                const int candidate_index = b + p * B_node;
                split_info_var[candidate_index] = feat_idx;
                split_info_bin_count_left[candidate_index] = cumul_count;
                split_info_bin_count_right[candidate_index] = n_sample - cumul_count;
                split_info_split_val[candidate_index] =
                    0.5f * (tw.bin_max[b] + tw.bin_min[b + 1]);

                const float left_count_m = float(cumul_count);
                const float right_count_m = float(n_sample - cumul_count);
                for (int t = 0; t < K; ++t) {
                    const float right_sum_t = overall_total_sum[t] - tw.left_sum[t];
                    const float sst_reduction =
                        (tw.left_sum[t] * tw.left_sum[t]) / left_count_m
                        + (right_sum_t * right_sum_t) / right_count_m
                        - critParent[t];
                    aggregated_objectives[candidate_index * n_objective
                                          + target_weight_map[t]]
                        += -sst_reduction * weights_target[t];
                }
            }
        }
    }

    if (unique_counts == n_predictor) {
        return {-1, -1, -1.0f};
    }

    // Select best.
    int best_index = 0;
    if (n_objective == 1) {
        float best_val = aggregated_objectives[0];
        for (int i = 1; i < n_cand; ++i) {
            if (aggregated_objectives[i * n_objective] < best_val) {
                best_val = aggregated_objectives[i * n_objective];
                best_index = i;
            }
        }
    } else if (agg_mode == 0) {
        // Rank aggregation: for each objective, write a stable rank into rank_obj,
        // then weighted-sum.
        for (int o = 0; o < n_objective; ++o) {
            // sort indices by aggregated_objectives[:, o] ascending stable
            for (int i = 0; i < n_cand; ++i) sort_idx_obj[i] = i;
            std::stable_sort(
                sort_idx_obj.begin(), sort_idx_obj.begin() + n_cand,
                [&](int a, int b) {
                    return aggregated_objectives[a * n_objective + o]
                         < aggregated_objectives[b * n_objective + o];
                });
            for (int k = 0; k < n_cand; ++k) {
                rank_obj[sort_idx_obj[k] * n_objective + o] = k;
            }
        }
        for (int i = 0; i < n_cand; ++i) {
            float s = 0.0f;
            for (int o = 0; o < n_objective; ++o) {
                s += float(rank_obj[i * n_objective + o]) * objective_weights[o];
            }
            objective_weighted_rank[i] = s;
        }
        float best_val = objective_weighted_rank[0];
        for (int i = 1; i < n_cand; ++i) {
            if (objective_weighted_rank[i] < best_val) {
                best_val = objective_weighted_rank[i];
                best_index = i;
            }
        }
    } else {
        // Score aggregation (per-objective min-max normalization).
        constexpr float EPS = 1e-12f;
        for (int i = 0; i < n_cand; ++i) objective_weighted_rank[i] = 0.0f;
        for (int o = 0; o < n_objective; ++o) {
            float obj_min = aggregated_objectives[0 * n_objective + o];
            float obj_max = obj_min;
            for (int i = 1; i < n_cand; ++i) {
                const float v = aggregated_objectives[i * n_objective + o];
                if (v < obj_min) obj_min = v;
                if (v > obj_max) obj_max = v;
            }
            const float obj_range = obj_max - obj_min;
            if (obj_range < EPS) continue;
            for (int i = 0; i < n_cand; ++i) {
                objective_weighted_rank[i] += objective_weights[o]
                    * ((aggregated_objectives[i * n_objective + o] - obj_min) / obj_range);
            }
        }
        float best_val = objective_weighted_rank[0];
        for (int i = 1; i < n_cand; ++i) {
            if (objective_weighted_rank[i] < best_val) {
                best_val = objective_weighted_rank[i];
                best_index = i;
            }
        }
    }

    if (split_info_var[best_index] == -1) {
        return {-1, -1, -1.0f};
    }

    if (min_gain_to_split > 0.0f) {
        float gain_num = 0.0f, wsum = 0.0f;
        for (int o = 0; o < n_objective; ++o) {
            float between_o = -aggregated_objectives[best_index * n_objective + o];
            float within_o = 0.0f;
            for (int t = 0; t < K; ++t)
                if (target_weight_map[t] == o)
                    within_o += weights_target[t] * (sumsq[t] - critParent[t]);
            float rel = (within_o > 1e-12f) ? (between_o / within_o) : 0.0f;
            if (rel < 0.0f) rel = 0.0f;
            if (rel > 1.0f) rel = 1.0f;
            gain_num += objective_weights[o] * rel;
            wsum += objective_weights[o];
        }
        float gain = (wsum > 1e-12f) ? (gain_num / wsum) : 0.0f;
        if (gain < min_gain_to_split) return {-1, -1, -1.0f};  // prune: weak split
    }

    const int split_var = split_info_var[best_index];
    const float split_val = split_info_split_val[best_index];
    const int left_count_m = split_info_bin_count_left[best_index];

    // SORT row_indices[ndstart:ndend] by features[row, split_var] ascending,
    // matching cfuncs_fast's M1 stable_argsort_1d_f32 + reorder pattern.
    // This is critical for matching cfuncs_fast's deeper-node accumulations.
    // sprint-6: packed-key sort. Follow-up: gather the split-feature values
    // into a contiguous buffer and route through packed_key_argsort_f32, the
    // SAME hybrid (std::sort small-n / stable LSD radix large-n) used by the
    // per-feature scan. Bit-identical permutation to the inline packed
    // std::sort (same fp32_order_key, same (key asc, index asc) order), and
    // large nodes get the radix speedup here too. Runs post-parallel-region
    // on the calling thread, so its tw_global scratch is safe to reuse.
    sort_scratch.resize(2 * n_sample);
    int * idx = sort_scratch.data();           // permutation of [0, n_sample)
    int * tmp = sort_scratch.data() + n_sample;
    const int sv = split_var;
    {
        ScanWorkspace& tw = tw_global;
        tw.temp_x.resize(n_sample);
        tw.qb_keys.resize(n_sample);
        for (int i = 0; i < n_sample; ++i) {
            tw.temp_x[i] = features[row_indices[ndstart + i] * P + sv];
        }
        packed_key_argsort_f32(tw.temp_x.data(), n_sample,
                               tw.qb_keys.data(), idx);
    }
    for (int i = 0; i < n_sample; ++i) {
        tmp[i] = row_indices[ndstart + idx[i]];
    }
    for (int i = 0; i < n_sample; ++i) {
        row_indices[ndstart + i] = tmp[i];
    }

    return {left_count_m, split_var, split_val};
}

}  // anonymous namespace


py::array_t<float> predict_mt_tree(
    py::array_t<const float, py::array::c_style | py::array::forcecast> features,
    py::array_t<const int, py::array::c_style | py::array::forcecast> node_stats,
    py::array_t<const float, py::array::c_style | py::array::forcecast> split_vals,
    py::array_t<const float, py::array::c_style | py::array::forcecast> node_means,
    int maxdepth,
    int /*verbose*/,
    int num_threads
) {
    const int N = static_cast<int>(features.shape(0));
    const int P = static_cast<int>(features.shape(1));
    const int num_nodes = static_cast<int>(node_stats.shape(0));
    const int K = static_cast<int>(node_means.shape(1));

    auto out = py::array_t<float>({N, K});
    float * out_ptr = out.mutable_data();
    const float * feat = features.data();
    const int * ns = node_stats.data();
    const float * sv = split_vals.data();
    const float * nm = node_means.data();

    // Each row is predicted independently (own tree walk, writes only its own
    // out_ptr[i*K..]; ns/sv/nm/feat are read-only). Parallelizing over rows is
    // bit-exact -- the per-row outputs are identical regardless of thread
    // assignment. mGBT calls predict O(trees) times (residual update + ES eval
    // + final recompute); mDT/mRF call it at forecast time. Previously serial.
    #pragma omp parallel for num_threads(num_threads) if (num_threads > 1) \
            schedule(static)
    for (int i = 0; i < N; ++i) {
        int node = 0;
        int depth = 0;
        while (depth < maxdepth) {
            const int status = ns[node * 7 + 2];
            const int left_child = ns[node * 7 + 5];
            const int right_child = ns[node * 7 + 6];
            // Terminal node: status != NODE_INTERIOR (-3) means no further split.
            if (status != NODE_INTERIOR) break;
            // Sanity guard against malformed trees.
            if (left_child == 0 && right_child == 0) break;
            if (left_child < 0 || right_child < 0
                || left_child >= num_nodes || right_child >= num_nodes) break;
            const int split_var = ns[node * 7 + 4];
            if (feat[i * P + split_var] < sv[node]) {
                node = left_child;
            } else {
                node = right_child;
            }
            depth = ns[node * 7 + 3];
        }
        for (int t = 0; t < K; ++t) {
            out_ptr[i * K + t] = nm[node * K + t];
        }
    }
    return out;
}

// =============================================================================
// fit_mt_tree_bin_impl
//
// Shared implementation for both fit_mt_tree_bin (contiguous, row_indices =
// 0..N-1) and fit_mt_tree_bin_indirect (positions-based row_indices into a
// larger base array). All feature/label accesses go through row_indices[],
// so the impl is oblivious to whether feat_ptr/lab_ptr point to a sampled
// subset (size N rows) or to the full base array (size >= N rows).
// =============================================================================
static py::tuple fit_mt_tree_bin_impl(
    const float * feat_ptr,
    const float * lab_ptr,
    int N, int P, int K,
    std::vector<int> && row_indices_in,
    py::array_t<float> weights_target,
    py::array_t<int> target_weight_map,
    int n_objective,
    py::array_t<float> objective_weights,
    int max_depth,
    float mtry,
    int samp_feat_by_node,
    int min_data_in_leaf,
    int num_nodes,
    int num_bins,
    int num_threads,
    int agg_mode,
    int obj_weight_mode,
    float obj_weight_dirichlet_alpha,
    float min_gain_to_split,
    const py::object & rng,
    int64_t cxx_rng_seed
) {
    // Feature-sampling RNG selection:
    //  - cxx_rng_seed >= 0: pure-C++ std::mt19937 (sample_features_cxx). Touches
    //    NO Python, so the whole node loop runs GIL-free -> the threading
    //    backend parallelises at full speed, and every method gets a cheaper
    //    per-node draw (no numpy call). NOT bit-exact to numpy; deterministic
    //    per tree + identical on Linux/Mac (mt19937 + manual modulo).
    //  - cxx_rng_seed < 0: numpy path -- per-tree RandomState `rng` (or global
    //    if None). Bit-exact to the legacy contract; GIL re-acquired per node.
    const bool use_cxx_rng = (cxx_rng_seed >= 0);
    std::mt19937 cxx_gen(static_cast<uint32_t>(use_cxx_rng ? cxx_rng_seed : 0));
    std::vector<int> cxx_pool(use_cxx_rng ? P : 0);
    const py::object feat_perm = (use_cxx_rng || rng.is_none())
        ? py::object(py::none()) : rng.attr("permutation");
    int n_features;
    if (mtry < 1.0f) {
        n_features = static_cast<int>(mtry * P) + 1;
    } else {
        n_features = P;
    }

    const int num_full_nodes = 2 * num_nodes - 1;

    // Allocate output arrays.
    auto node_stats = py::array_t<int>({num_full_nodes, 7});
    auto split_vals = py::array_t<float>(num_full_nodes);
    auto node_means = py::array_t<float>({num_full_nodes, K});

    int * ns = node_stats.mutable_data();
    float * sv = split_vals.mutable_data();
    float * nm = node_means.mutable_data();
    for (int i = 0; i < num_full_nodes * 7; ++i) ns[i] = 0;
    for (int i = 0; i < num_full_nodes; ++i) sv[i] = 0.0f;
    for (int i = 0; i < num_full_nodes * K; ++i) nm[i] = 0.0f;

    std::vector<int> node_starts(num_full_nodes, 0);
    std::vector<int> node_ends(num_full_nodes, 0);
    std::vector<int> row_indices = std::move(row_indices_in);

    // Root node.
    ns[0 * 7 + 0] = 0;
    ns[0 * 7 + 1] = N;
    ns[0 * 7 + 2] = NODE_TOSPLIT;
    ns[0 * 7 + 3] = 0;
    node_starts[0] = 0;
    node_ends[0] = N;
    {
        // Root mean over rows in row_indices (NOT 0..N-1) so the impl works
        // for both the contiguous and indirect entries.
        for (int t = 0; t < K; ++t) {
            float s = 0.0f;
            for (int i = 0; i < N; ++i) s += lab_ptr[row_indices[i] * K + t];
            nm[0 * K + t] = s / float(N);
        }
    }

    std::vector<int> feature_indices(n_features, 0);
    if (n_features == P) {
        for (int i = 0; i < P; ++i) feature_indices[i] = i;
    } else if (samp_feat_by_node == 0) {
        if (use_cxx_rng) {
            sample_features_cxx(cxx_gen, cxx_pool, P, n_features,
                                feature_indices.data());
        } else {
            sample_features_np(feat_perm, P, n_features, feature_indices.data());
        }
    }

    // Workspace. Per-feature scratch (temp_x, bin_rep, bin_min, bin_max,
    // bin_instance_count, total_sum, left_sum, qb_sort_idx) now lives
    // thread_local inside find_best_split_mod for M4 OpenMP safety.
    std::vector<float> critParent(K, 0.0f);
    std::vector<float> overall_total_sum(K, 0.0f);

    const int n_cand_total = n_features * num_bins;
    std::vector<float> aggregated_objectives(n_cand_total * n_objective, 0.0f);
    std::vector<float> objective_weighted_rank(n_cand_total, 0.0f);
    std::vector<int> sort_idx_obj(n_cand_total, 0);
    std::vector<int> rank_obj(n_cand_total * n_objective, 0);
    std::vector<int> split_info_var(n_cand_total, -1);
    std::vector<int> split_info_bin_count_left(n_cand_total, 0);
    std::vector<int> split_info_bin_count_right(n_cand_total, 0);
    std::vector<float> split_info_split_val(n_cand_total, 0.0f);
    std::vector<int> sort_scratch;

    // feat_ptr / lab_ptr are already passed in as function arguments.
    const float * wt_ptr = weights_target.data();
    const int * twm_ptr = target_weight_map.data();
    const float * objw_ptr = objective_weights.data();

    // Per-node objective-weight resampling buffer. Reused across nodes when
    // obj_weight_mode != FIXED so we don't allocate per node. When FIXED, we
    // pass the caller-supplied objw_ptr through unchanged (zero overhead, no
    // numpy callback) so the kernel is bit-exact to pre-change behavior.
    // OBJ_WEIGHT_MODE_FIXED=0 / DIRICHLET=1 / ONEHOT=2 (kept in sync with
    // CTree_MT.fit translation in mttrees/tree.py).
    std::vector<float> local_objw(
        obj_weight_mode == 0 ? 0 : n_objective, 0.0f);

    int current_node = 0;
    // Release the GIL for the ENTIRE node-building loop. The heavy work
    // (quantile_bins + per-feature scan + sort + row partition in
    // find_best_split_mod, plus the child-mean sums) is pure C++/OpenMP and
    // all py::array writes go through raw mutable_data() pointers obtained
    // above the loop -- GIL-free. We re-acquire the GIL ONLY for the brief
    // per-node numpy RNG callbacks (feature / objective sampling). Releasing
    // once (not per node) avoids GIL ping-pong; this is what lets the joblib
    // THREADING backend run trees concurrently. Single-threaded / loky callers
    // just release an uncontended GIL (~free). Bit-exact: loop is deterministic.
    {
    py::gil_scoped_release gil_release;
    for (int k = 0; k < num_full_nodes - 2; ++k) {
        if (k > current_node || current_node >= num_full_nodes - 2) break;
        if (ns[k * 7 + 2] != NODE_TOSPLIT) continue;

        if (samp_feat_by_node == 1) {
            if (use_cxx_rng) {
                // GIL-free: no Python touched -> threads run fully concurrently.
                sample_features_cxx(cxx_gen, cxx_pool, P, n_features,
                                    feature_indices.data());
            } else {
                py::gil_scoped_acquire gil_acquire;
                sample_features_np(feat_perm, P, n_features, feature_indices.data());
            }
        }

        // Resolve the per-node objective-weight pointer. For FIXED, this is
        // the caller-supplied vector (same pointer at every node). For
        // DIRICHLET / ONEHOT, we resample into local_objw before each split
        // search.
        const float * objw_ptr_dyn = objw_ptr;
        if (obj_weight_mode == 1) {
            py::gil_scoped_acquire gil_acquire;
            sample_dirichlet_via_np(
                rng, n_objective, obj_weight_dirichlet_alpha, local_objw.data());
            objw_ptr_dyn = local_objw.data();
        } else if (obj_weight_mode == 2) {
            int pick;
            {
                py::gil_scoped_acquire gil_acquire;
                pick = sample_categorical_via_np(rng, n_objective);
            }
            std::fill(local_objw.begin(), local_objw.end(), 0.0f);
            local_objw[pick] = 1.0f;
            objw_ptr_dyn = local_objw.data();
        }

        // GIL already released at loop level; find_best_split_mod is pure
        // C++/OpenMP (the ~80% hot path) and runs concurrently across threads.
        BestSplit res = find_best_split_mod(
            feat_ptr, N, P, lab_ptr, K,
            row_indices.data(), ns[k * 7 + 1], n_features, n_objective,
            node_starts[k], node_ends[k],
            feature_indices.data(), min_data_in_leaf,
            wt_ptr, twm_ptr, objw_ptr_dyn,
            num_bins, agg_mode, num_threads, min_gain_to_split,
            critParent, overall_total_sum,
            aggregated_objectives, objective_weighted_rank,
            sort_idx_obj, rank_obj,
            split_info_var, split_info_bin_count_left,
            split_info_bin_count_right, split_info_split_val,
            sort_scratch);

        if (res.split_var == -1) {
            ns[k * 7 + 2] = NODE_TERMINAL;
        } else {
            const int last_index = res.last_index;
            ns[k * 7 + 2] = NODE_INTERIOR;
            ns[k * 7 + 4] = res.split_var;
            sv[k] = res.split_val;
            ns[k * 7 + 5] = current_node + 1;
            ns[k * 7 + 6] = current_node + 2;
            ns[(current_node + 1) * 7 + 0] = current_node + 1;
            ns[(current_node + 2) * 7 + 0] = current_node + 2;
            ns[(current_node + 1) * 7 + 1] = last_index;
            ns[(current_node + 2) * 7 + 1] = ns[k * 7 + 1] - last_index;
            ns[(current_node + 1) * 7 + 3] = ns[k * 7 + 3] + 1;
            ns[(current_node + 2) * 7 + 3] = ns[k * 7 + 3] + 1;

            if (ns[(current_node + 1) * 7 + 1] < 2 * min_data_in_leaf
                || ns[(current_node + 1) * 7 + 3] == max_depth) {
                ns[(current_node + 1) * 7 + 2] = NODE_TERMINAL;
            } else {
                ns[(current_node + 1) * 7 + 2] = NODE_TOSPLIT;
            }
            if (ns[(current_node + 2) * 7 + 1] < 2 * min_data_in_leaf
                || ns[(current_node + 2) * 7 + 3] == max_depth) {
                ns[(current_node + 2) * 7 + 2] = NODE_TERMINAL;
            } else {
                ns[(current_node + 2) * 7 + 2] = NODE_TOSPLIT;
            }
            node_starts[current_node + 1] = node_starts[k];
            node_ends[current_node + 1] = node_starts[k] + last_index;
            node_starts[current_node + 2] = node_ends[current_node + 1];
            node_ends[current_node + 2] = node_ends[k];

            // Child node means (from sums over partitioned row_indices).
            for (int t = 0; t < K; ++t) {
                float sl = 0.0f;
                float sr = 0.0f;
                for (int i = 0; i < last_index; ++i) {
                    sl += lab_ptr[row_indices[node_starts[k] + i] * K + t];
                }
                for (int i = last_index; i < ns[k * 7 + 1]; ++i) {
                    sr += lab_ptr[row_indices[node_starts[k] + i] * K + t];
                }
                nm[(current_node + 1) * K + t] = sl / float(last_index);
                nm[(current_node + 2) * K + t] = sr / float(ns[k * 7 + 1] - last_index);
            }

            current_node += 2;
        }
    }
    }  // end gil_scoped_release: re-acquire GIL for the py::array output below

    const int actual_nodes = current_node + 1;
    for (int k = 0; k < actual_nodes; ++k) {
        if (ns[k * 7 + 2] == NODE_TOSPLIT) ns[k * 7 + 2] = NODE_TERMINAL;
    }

    // Truncate outputs to actual_nodes by allocating new arrays of that size.
    auto out_ts = py::array_t<int>({actual_nodes, 7});
    auto out_sv = py::array_t<float>(actual_nodes);
    auto out_nm = py::array_t<float>({actual_nodes, K});
    int * out_ts_ptr = out_ts.mutable_data();
    float * out_sv_ptr = out_sv.mutable_data();
    float * out_nm_ptr = out_nm.mutable_data();
    for (int k = 0; k < actual_nodes; ++k) {
        for (int j = 0; j < 7; ++j) {
            out_ts_ptr[k * 7 + j] = ns[k * 7 + j];
        }
        out_sv_ptr[k] = sv[k];
        for (int t = 0; t < K; ++t) {
            out_nm_ptr[k * K + t] = nm[k * K + t];
        }
    }

    // Adaptive shrink. Free the per-thread scratch buffers when the
    // previously-held capacity is much larger than this fit's root size.
    // Hot path (capacity already close to N): O(1) check, no realloc.
    // Cross-dataset boundary (capacity from prior larger N): one free.
    // Runs across the same OMP threads that grew the workspace. Exercised
    // by tests/test_phase2.py::test_scanworkspace_shrinks_after_large_then_small_fit.
    #pragma omp parallel num_threads(num_threads) if (num_threads > 1)
    {
        tw_global.shrink_if_oversized(N);
    }

    return py::make_tuple(out_ts, out_sv, out_nm);
}


// =============================================================================
// Public entry: contiguous (pre-gathered) features/labels. Backward-compatible.
// =============================================================================
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
    int /*verbose*/,
    int agg_mode,
    int obj_weight_mode,
    float obj_weight_dirichlet_alpha,
    py::object rng,
    int64_t cxx_rng_seed
) {
    const int N = static_cast<int>(features.shape(0));
    const int P = static_cast<int>(features.shape(1));
    const int K = static_cast<int>(labels.shape(1));
    std::vector<int> row_indices(N);
    for (int i = 0; i < N; ++i) row_indices[i] = i;
    return fit_mt_tree_bin_impl(
        features.data(), labels.data(), N, P, K, std::move(row_indices),
        weights_target, target_weight_map, n_objective, objective_weights,
        max_depth, mtry, samp_feat_by_node, min_data_in_leaf, num_nodes,
        num_bins, num_threads, agg_mode,
        obj_weight_mode, obj_weight_dirichlet_alpha, min_gain_to_split,
        rng, cxx_rng_seed);
}


// =============================================================================
// Public entry: indirected access via positions[].
// True Cycle 2: no internal gather. row_indices[i] = positions[i] points
// directly into features_base / labels_base, so the kernel reads from the
// shared memmap by row position. Per-worker memory drops from ~315 MB to
// ~6 MB on Rossmann (n_jobs=10, bagging_fraction=0.8).
//
// Bit-exact to fit_mt_tree_bin(features_base[positions],
//                              labels_base[positions], ...) under the same
// global RNG state. Gated by tests/test_cpp_fit_indirect.py.
// =============================================================================
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
    int /*verbose*/,
    int agg_mode,
    int obj_weight_mode,
    float obj_weight_dirichlet_alpha,
    py::object rng,
    int64_t cxx_rng_seed
) {
    const int N = static_cast<int>(positions.shape(0));
    const int P = static_cast<int>(features_base.shape(1));
    const int K = static_cast<int>(labels_base.shape(1));
    std::vector<int> row_indices(N);
    const int * pos = positions.data();
    for (int i = 0; i < N; ++i) row_indices[i] = pos[i];
    return fit_mt_tree_bin_impl(
        features_base.data(), labels_base.data(), N, P, K,
        std::move(row_indices),
        weights_target, target_weight_map, n_objective, objective_weights,
        max_depth, mtry, samp_feat_by_node, min_data_in_leaf, num_nodes,
        num_bins, num_threads, agg_mode,
        obj_weight_mode, obj_weight_dirichlet_alpha, min_gain_to_split,
        rng, cxx_rng_seed);
}

// Test-only introspection: total capacity (bytes) of the per-thread scan
// workspace owned by the calling thread. Tests use this to verify the
// adaptive shrink at the end of fit_mt_tree_bin actually fires when a
// large fit is followed by a smaller one.
size_t ws_capacity_bytes() {
    return tw_global.capacity_bytes();
}

}  // namespace mttrees::cpp_kernel
