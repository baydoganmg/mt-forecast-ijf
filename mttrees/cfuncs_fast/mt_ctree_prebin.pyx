# distutils: language = c++
# cython: language_level=3
"""
Pre-binned multi-target tree: bins all features globally once
(O(N log N * P), one sort per feature) before tree building,
eliminating the per-node O(N log N) argsort at every node.
Also uses O(N) partition instead of O(N log N) argsort after split selection.
When n_objective == 1, skips the double-argsort ranking entirely.
"""
import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import parallel, prange
from libc.math cimport abs
import time

cdef int NODE_TERMINAL = -1
cdef int NODE_TOSPLIT = -2
cdef int NODE_INTERIOR = -3
cdef double RELATIVE_THRESHOLD = 0.0000001

cdef int[:] sample_features(int P, int n_features):
    if P < n_features:
        raise ValueError("P should be greater than n_features.")
    cdef int[:] result_view
    result = np.random.choice(range(P), size=n_features, replace=False).astype(np.intc)
    result_view = result
    return result_view


# ---------------------------------------------------------------------------
# Global pre-binning (quantile-based, O(N log N) per feature, done once)
# ---------------------------------------------------------------------------
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef precompute_global_bins(double[:, ::1] features, int n_bins):
    """Bin every feature globally using equal-frequency (quantile) bins.

    Each feature is sorted once (O(N log N)), then split into ~equal-count
    bins.  Total cost: O(N log N * P), done once before tree building.

    Returns
    -------
    global_bins      : int[N, P]       – bin id for each (sample, feature)
    global_bin_max   : double[P, n_bins] – max observed value per bin
    global_bin_min   : double[P, n_bins] – min observed value per bin
    global_is_unique : int[P]          – 1 when feature is constant
    """
    cdef int N = features.shape[0]
    cdef int P = features.shape[1]
    cdef int i, p, b
    cdef int current_bin_id, current_bin_count, approximate_bin_count, actual_bins
    cdef double NEG_INF = -1e308
    cdef double POS_INF = 1e308
    cdef double eps = 1e-10

    cdef short[:, ::1] g_bins = np.zeros((N, P), dtype=np.int16)
    cdef double[:, ::1] g_bin_max = np.full((P, n_bins), NEG_INF, dtype=np.double)
    cdef double[:, ::1] g_bin_min = np.full((P, n_bins), POS_INF, dtype=np.double)
    cdef int[:] g_is_unique = np.zeros(P, dtype=np.intc)

    # temporary per-feature buffer
    cdef double[:] col = np.zeros(N, dtype=np.double)
    cdef int[:] si = np.zeros(N, dtype=np.intc)

    for p in range(P):
        # extract column
        for i in range(N):
            col[i] = features[i, p]

        # sort indices (one np.argsort per feature, done once globally)
        si = np.argsort(np.asarray(col)).astype(np.intc)

        # check uniqueness
        if col[si[N - 1]] - col[si[0]] < RELATIVE_THRESHOLD:
            g_is_unique[p] = 1
            continue

        # equal-frequency binning (same logic as original quantile_bins)
        actual_bins = n_bins
        if actual_bins > N:
            actual_bins = N
        approximate_bin_count = N // actual_bins

        current_bin_id = 0
        current_bin_count = 1
        g_bins[si[0], p] = 0
        g_bin_max[p, 0] = col[si[0]]
        g_bin_min[p, 0] = col[si[0]]

        for i in range(1, N):
            # new bin when current is full AND value differs from previous
            if col[si[i - 1]] + RELATIVE_THRESHOLD < col[si[i]]:
                if current_bin_count >= approximate_bin_count:
                    current_bin_count = 0
                    current_bin_id += 1

            if current_bin_id >= actual_bins:
                current_bin_id = actual_bins - 1

            g_bins[si[i], p] = current_bin_id
            current_bin_count += 1

            if col[si[i]] > g_bin_max[p, current_bin_id]:
                g_bin_max[p, current_bin_id] = col[si[i]]
            if col[si[i]] < g_bin_min[p, current_bin_id]:
                g_bin_min[p, current_bin_id] = col[si[i]]

        if current_bin_id == 0:
            g_is_unique[p] = 1
            continue

        # propagate boundaries through trailing empty bins
        for b in range(current_bin_id + 1, n_bins):
            g_bin_min[p, b] = g_bin_max[p, current_bin_id]
            g_bin_max[p, b] = g_bin_max[p, current_bin_id]

    return (np.asarray(g_bins), np.asarray(g_bin_max),
            np.asarray(g_bin_min), np.asarray(g_is_unique))


# ---------------------------------------------------------------------------
# Tree fitting with pre-computed bins
# ---------------------------------------------------------------------------
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef fit_mt_tree_prebin(
        double[:, ::1] features, double[:, ::1] labels,
        double[:] weights_target, int[:] target_weight_map,
        int n_objective, double[:] objective_weights,
        int max_depth, double mtry, int samp_feat_by_node,
        int min_data_in_leaf, int num_nodes, double min_gain_to_split,
        int num_bins, int num_threads, int verbose,
        # --- pre-computed bin data ---
        short[:, ::1] global_bins, double[:, ::1] global_bin_max,
        double[:, ::1] global_bin_min, int[:] global_is_unique,
        # --- objective aggregation mode: 0 = rank, 1 = score (min-max) ---
        int agg_mode):

    cdef int N = features.shape[0]
    cdef int P = features.shape[1]
    cdef int N_t = labels.shape[1]
    cdef int i, j, current_node, k, num_full_nodes, split_var, n_features
    cdef double split_val

    num_full_nodes = 2 * num_nodes - 1

    cdef double[:, ::1] node_means = np.zeros((num_full_nodes, N_t), dtype=np.double)
    cdef int[:, ::1] node_stats = np.zeros((num_full_nodes, 7), dtype=np.intc)
    cdef double[:] split_vals = np.zeros(num_full_nodes, dtype=np.double)
    cdef int[:] node_starts = np.zeros(num_full_nodes, dtype=np.intc)
    cdef int[:] node_ends = np.zeros(num_full_nodes, dtype=np.intc)
    cdef int[:] row_indices = np.zeros(N, dtype=np.intc)

    if mtry < 1:
        n_features = int(mtry * P) + 1
    else:
        n_features = P

    cdef int[:] feature_indices = np.zeros(n_features, dtype=np.intc)

    # workspace arrays for split evaluation
    cdef double[:, :, :] split_perf = np.zeros((n_features, num_bins, N_t), dtype=np.double)
    cdef double[:, :, :] split_info = np.full((n_features, num_bins, 5), np.nan, dtype=np.double)
    cdef double[:, ::1] total_sum = np.zeros((num_bins, N_t), dtype=np.double)
    cdef double[:] overall_total_sum = np.zeros(N_t, dtype=np.double)
    cdef double[:, :] aggregated_objectives = np.zeros((n_features * num_bins, n_objective), dtype=np.double)
    cdef double[:] objective_weighted_rank = np.zeros(n_features * num_bins, dtype=np.double)
    cdef int[:, :] sort_idx_obj = np.zeros((n_features * num_bins, n_objective), dtype=np.intc)
    cdef int[:, :] rank_obj = np.zeros((n_features * num_bins, n_objective), dtype=np.intc)
    cdef double[:] left_sum = np.zeros(N_t, dtype=np.double)
    cdef double[:] right_sum = np.zeros(N_t, dtype=np.double)
    cdef double[:] critParent = np.zeros(N_t, dtype=np.double)
    cdef int[:] node_bin_counts = np.zeros(num_bins, dtype=np.intc)

    # partition buffer (reused across nodes)
    cdef int[:] sorted_row_indices = np.zeros(N, dtype=np.intc)

    train_start = time.time()
    for i in range(N):
        row_indices[i] = i

    # root node
    node_stats[0, 0] = 0
    node_stats[0, 1] = N
    node_stats[0, 2] = NODE_TOSPLIT
    node_stats[0, 3] = 0
    node_starts[0] = 0
    node_ends[0] = N

    for j in range(N_t):
        node_means[0, j] = 0
        for i in range(N):
            node_means[0, j] += labels[i, j]
        node_means[0, j] = node_means[0, j] / N

    # feature sampling (no per-node resampling case)
    if n_features == P:
        for i in range(P):
            feature_indices[i] = i
    elif samp_feat_by_node == 0:
        feature_indices = sample_features(P, n_features)

    current_node = 0
    for k in range(num_full_nodes - 2):
        if k > current_node or current_node >= num_full_nodes - 2:
            break
        if node_stats[k, 2] != NODE_TOSPLIT:
            continue

        if samp_feat_by_node == 1:
            feature_indices = sample_features(P, n_features)

        last_index, split_var, split_val = find_best_split_prebin(
            features, labels, row_indices, node_stats[k, 1], N_t,
            n_features, n_objective,
            node_starts[k], node_ends[k], feature_indices, min_data_in_leaf,
            split_perf, split_info, total_sum, overall_total_sum,
            weights_target, target_weight_map,
            aggregated_objectives, sort_idx_obj, rank_obj,
            objective_weights, objective_weighted_rank,
            num_threads, num_bins, verbose,
            left_sum, right_sum, critParent,
            node_bin_counts, sorted_row_indices,
            global_bins, global_bin_max, global_bin_min, global_is_unique,
            agg_mode)

        if split_var == -1:
            node_stats[k, 2] = NODE_TERMINAL
        else:
            node_stats[k, 2] = NODE_INTERIOR
            node_stats[k, 4] = split_var
            split_vals[k] = split_val
            node_stats[k, 5] = current_node + 1
            node_stats[k, 6] = current_node + 2
            node_stats[current_node + 1, 0] = current_node + 1
            node_stats[current_node + 2, 0] = current_node + 2
            node_stats[current_node + 1, 1] = last_index
            node_stats[current_node + 2, 1] = node_stats[k, 1] - last_index
            node_stats[current_node + 1, 3] = node_stats[k, 3] + 1
            node_stats[current_node + 2, 3] = node_stats[k, 3] + 1

            if node_stats[current_node + 1, 1] < 2 * min_data_in_leaf or node_stats[current_node + 1, 3] == max_depth:
                node_stats[current_node + 1, 2] = NODE_TERMINAL
            else:
                node_stats[current_node + 1, 2] = NODE_TOSPLIT

            if node_stats[current_node + 2, 1] < 2 * min_data_in_leaf or node_stats[current_node + 2, 3] == max_depth:
                node_stats[current_node + 2, 2] = NODE_TERMINAL
            else:
                node_stats[current_node + 2, 2] = NODE_TOSPLIT

            node_starts[current_node + 1] = node_starts[k]
            node_ends[current_node + 1] = node_starts[k] + last_index
            node_starts[current_node + 2] = node_ends[current_node + 1]
            node_ends[current_node + 2] = node_ends[k]

            # child node means (from sums accumulated during partition)
            for j in range(N_t):
                node_means[current_node + 1, j] = left_sum[j] / last_index
                node_means[current_node + 2, j] = right_sum[j] / (node_stats[k, 1] - last_index)

            current_node = current_node + 2

    cdef int actual_nodes = current_node + 1
    for k in range(actual_nodes):
        if node_stats[k, 2] == NODE_TOSPLIT:
            node_stats[k, 2] = NODE_TERMINAL

    if verbose == 1:
        t2 = time.time()
        print('Tree fitting time: ', t2 - train_start)

    return (np.asarray(node_stats)[:actual_nodes].copy(),
            np.asarray(split_vals)[:actual_nodes].copy(),
            np.asarray(node_means)[:actual_nodes].copy())


# ---------------------------------------------------------------------------
# Split finding with pre-computed bins (no per-node sorting)
# ---------------------------------------------------------------------------
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef find_best_split_prebin(
        double[:, ::1] features, double[:, ::1] labels,
        int[:] row_indices, int n_sample, int n_target,
        int n_predictor, int n_objective,
        int ndstart, int ndend, int[:] feature_indices,
        int min_data_in_leaf,
        double[:, :, :] split_perf, double[:, :, :] split_info,
        double[:, :] total_sum, double[:] overall_total_sum,
        double[:] weights_target, int[:] target_weight_map,
        double[:, :] aggregated_objectives,
        int[:, :] sort_idx_obj, int[:, :] rank_obj,
        double[:] objective_weights, double[:] objective_weighted_rank,
        int num_threads, int given_bin_count, int verbose,
        double[:] left_sum, double[:] right_sum, double[:] critParent,
        int[:] node_bin_counts, int[:] sorted_row_indices,
        # --- global bin data ---
        short[:, ::1] global_bins, double[:, ::1] global_bin_max,
        double[:, ::1] global_bin_min, int[:] global_is_unique,
        # --- aggregation mode: 0 = rank, 1 = score (min-max) ---
        int agg_mode):

    cdef double sst_reduction
    cdef int i, t, b, o, p, feat_idx
    cdef int left_count_m, right_count_m, cumul_count
    cdef int unique_counts, n_occupied
    cdef int split_var, best_index, split_val_index, split_var_index
    cdef double best_val
    cdef double split_val
    cdef int left_pos, right_pos
    cdef int n_cand
    cdef double obj_min, obj_max, obj_range, v
    cdef double EPS = 1e-12

    # parent loss
    for t in range(n_target):
        overall_total_sum[t] = 0
    for i in range(ndstart, ndend):
        for t in range(n_target):
            overall_total_sum[t] += labels[row_indices[i], t]
    for t in range(n_target):
        critParent[t] = overall_total_sum[t] * overall_total_sum[t] / n_sample

    # reset workspace
    for p in range(n_predictor):
        for b in range(given_bin_count):
            objective_weighted_rank[b + p * given_bin_count] = 0
            for o in range(n_objective):
                aggregated_objectives[b + p * given_bin_count, o] = 0

    unique_counts = 0

    for p in range(n_predictor):
        feat_idx = feature_indices[p]

        # mark all bins as "no split" initially
        for b in range(given_bin_count):
            split_info[p, b, 0] = -1
            for t in range(n_target):
                total_sum[b, t] = 0.0
            node_bin_counts[b] = 0

        # skip globally-unique features
        if global_is_unique[feat_idx] == 1:
            unique_counts += 1
            continue

        # accumulate per-bin sums and counts for this node  (O(n_node))
        for i in range(ndstart, ndend):
            b = global_bins[row_indices[i], feat_idx]
            node_bin_counts[b] += 1
            for t in range(n_target):
                total_sum[b, t] += labels[row_indices[i], t]

        # check node-level uniqueness
        n_occupied = 0
        for b in range(given_bin_count):
            if node_bin_counts[b] > 0:
                n_occupied += 1
        if n_occupied <= 1:
            unique_counts += 1
            continue

        # evaluate candidate splits  (O(n_bins * n_target))
        for t in range(n_target):
            left_sum[t] = 0
        cumul_count = 0

        for b in range(given_bin_count - 1):
            cumul_count += node_bin_counts[b]
            for t in range(n_target):
                left_sum[t] += total_sum[b, t]

            # not enough samples on left yet
            if cumul_count < min_data_in_leaf:
                if cumul_count + node_bin_counts[b + 1] < n_sample:
                    continue

            # right side too small — stop (prevents division by zero
            # when cumul_count == n_sample past the last occupied node bin)
            if n_sample - cumul_count < min_data_in_leaf:
                break

            split_info[p, b, 0] = feat_idx
            split_info[p, b, 1] = b
            split_info[p, b, 2] = (global_bin_max[feat_idx, b] + global_bin_min[feat_idx, b + 1]) / 2
            split_info[p, b, 3] = cumul_count
            split_info[p, b, 4] = n_sample - cumul_count

            for t in range(n_target):
                right_sum[t] = overall_total_sum[t] - left_sum[t]
                left_count_m = cumul_count
                right_count_m = n_sample - cumul_count
                sst_reduction = ((left_sum[t] * left_sum[t]) / left_count_m
                                 + (right_sum[t] * right_sum[t]) / right_count_m
                                 - critParent[t])
                split_perf[p, b, t] = sst_reduction
                aggregated_objectives[b + p * given_bin_count, target_weight_map[t]] += (
                    -sst_reduction * weights_target[t])

    if unique_counts == n_predictor:
        return -1, -1, -1

    # ---- select best split ----
    if n_objective == 1:
        # fast path: direct argmin, no ranking needed
        best_val = aggregated_objectives[0, 0]
        best_index = 0
        for i in range(1, n_predictor * given_bin_count):
            if aggregated_objectives[i, 0] < best_val:
                best_val = aggregated_objectives[i, 0]
                best_index = i
    elif agg_mode == 0:
        # multi-objective rank aggregation: double argsort, weighted-sum-of-ranks
        sort_idx_obj = np.argsort(np.asarray(aggregated_objectives), kind='stable', axis=0).astype(np.intc)
        rank_obj = np.argsort(np.asarray(sort_idx_obj), kind='stable', axis=0).astype(np.intc)
        for p in range(n_predictor):
            for b in range(given_bin_count):
                for o in range(n_objective):
                    objective_weighted_rank[b + p * given_bin_count] += (
                        rank_obj[b + p * given_bin_count, o] * objective_weights[o])
        best_index = np.argmin(np.asarray(objective_weighted_rank)).astype(np.intc)
    else:
        # multi-objective score aggregation: per-objective min-max normalization,
        # weighted-sum-of-normalized-scores, then argmin (pure Cython, no numpy)
        n_cand = n_predictor * given_bin_count
        for i in range(n_cand):
            objective_weighted_rank[i] = 0.0
        for o in range(n_objective):
            obj_min = aggregated_objectives[0, o]
            obj_max = aggregated_objectives[0, o]
            for i in range(1, n_cand):
                v = aggregated_objectives[i, o]
                if v < obj_min:
                    obj_min = v
                if v > obj_max:
                    obj_max = v
            obj_range = obj_max - obj_min
            if obj_range < EPS:
                continue
            for i in range(n_cand):
                objective_weighted_rank[i] += objective_weights[o] * (
                    (aggregated_objectives[i, o] - obj_min) / obj_range)
        best_val = objective_weighted_rank[0]
        best_index = 0
        for i in range(1, n_cand):
            if objective_weighted_rank[i] < best_val:
                best_val = objective_weighted_rank[i]
                best_index = i

    split_val_index = best_index % given_bin_count
    split_var_index = best_index // given_bin_count
    split_var = <int>(split_info[split_var_index, split_val_index, 0])
    split_val = split_info[split_var_index, split_val_index, 2]

    if split_info[split_var_index, split_val_index, 0] == -1:
        return -1, -1, -1

    # ---- O(n) partition + child sum accumulation (2 passes) ----
    # Pass 1: partition into buffer (left from front, right from back)
    #         and accumulate label sums for child means
    for t in range(n_target):
        left_sum[t] = 0.0
        right_sum[t] = 0.0
    left_pos = 0
    right_pos = n_sample - 1
    cdef int ri
    for i in range(n_sample):
        ri = row_indices[ndstart + i]
        if features[ri, split_var] < split_val:
            sorted_row_indices[left_pos] = ri
            left_pos += 1
            for t in range(n_target):
                left_sum[t] += labels[ri, t]
        else:
            sorted_row_indices[right_pos] = ri
            right_pos -= 1
            for t in range(n_target):
                right_sum[t] += labels[ri, t]

    left_count_m = left_pos

    # Pass 2: copy back
    for i in range(n_sample):
        row_indices[ndstart + i] = sorted_row_indices[i]

    if left_count_m == 0 or left_count_m == n_sample:
        return -1, -1, -1

    return left_count_m, split_var, split_val


# ---------------------------------------------------------------------------
# Prediction (identical to original)
# ---------------------------------------------------------------------------
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef predict_mt_tree(double[:, :] features, int[:, :] node_stats,
                      double[:] split_vals, double[:, :] node_means,
                      int maxdepth, int verbose, int num_threads=1):
    cdef int N = features.shape[0]
    cdef int N_t = node_means.shape[1]
    cdef double[:, :] prediction = np.zeros((N, N_t), dtype=np.double)
    cdef int i, k, t, split_var
    cdef double split_val
    cdef int _nthreads = num_threads

    for i in prange(N, nogil=True, num_threads=_nthreads):
        k = 0
        while node_stats[k, 2] != NODE_TERMINAL and node_stats[k, 3] < maxdepth:
            split_var = node_stats[k, 4]
            split_val = split_vals[k]
            if features[i, split_var] < split_val:
                k = node_stats[k, 5]
            else:
                k = node_stats[k, 6]
        for t in range(N_t):
            prediction[i, t] = node_means[k, t]

    return np.asarray(prediction)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef predict_mt_tree_node(double[:, :] features, int[:, :] node_stats,
                           double[:] split_vals, double[:, :] node_means,
                           int maxdepth, int verbose, int num_threads=1):
    cdef int N = features.shape[0]
    cdef int[:] prediction = np.zeros(N, dtype=np.intc)
    cdef int i, k, split_var
    cdef double split_val
    cdef int _nthreads = num_threads

    for i in prange(N, nogil=True, num_threads=_nthreads):
        k = 0
        while node_stats[k, 2] != NODE_TERMINAL and node_stats[k, 3] < maxdepth:
            split_var = node_stats[k, 4]
            split_val = split_vals[k]
            if features[i, split_var] < split_val:
                k = node_stats[k, 5]
            else:
                k = node_stats[k, 6]
        prediction[i] = k

    return np.asarray(prediction)
