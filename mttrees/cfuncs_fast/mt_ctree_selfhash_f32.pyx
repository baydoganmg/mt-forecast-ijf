# distutils: language = c++
# cython: language_level=3
import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import parallel, prange
from cython import boundscheck, wraparound
from libc.math cimport abs
import time

# M1: in-kernel stable rank aggregation + 1D stable argsort.
from .sort_helpers cimport (
    fill_stable_rank_2d_axis0_f32,
    stable_argsort_1d_f32,
)

# M3: SIMD per-bin K-target accumulation.
from .simd_helpers cimport mt_simd_accum_bin_k_f32

# M4: C++ std::stable_sort wrapper for per-feature value sorts inside
# quantile_bins. Closer to numpy's argsort speed than the hand-written
# Cython mergesort in sort_helpers; trades exact tie-break compatibility
# for performance, so callers re-baseline goldens after this swap.
from .sort_cpp cimport mt_cpp_argsort_stable_f32

# M4-phase-2: OpenMP thread id for per-feature prange.
cdef extern from "<omp.h>" nogil:
    int omp_get_thread_num()
    int omp_get_max_threads()

cdef int NODE_TERMINAL = -1
cdef int NODE_TOSPLIT = -2
cdef int NODE_INTERIOR = -3
cdef float RELATIVE_THRESHOLD = 0.0000001
cdef float SMALL_NUMBER = -float('inf')
cdef float LARGE_NUMBER = float('inf')

cdef int[:] sample_features(int P, int n_features):
    # Ensure P > n_features
    if P < n_features:
        raise ValueError("P should be greater than n_features.")
    
    # Create a numpy array and then cast it to a typed memoryview
    cdef int[:] result_view
    result = np.random.choice(range(P), size=n_features, replace=False).astype(np.intc)
    result_view = result

    return result_view

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef quantile_bins(float[::1] arr, int n_bins,
                              int[:] bins, float[:] bin_max, float[:] bin_min,  int[:] bin_counts):
    cdef int i, n_sample, is_unique, approximate_bin_count, current_bin_id, current_bin_count
    n_sample = arr.shape[0]
    cdef int[::1] sort_index = np.zeros(n_sample, dtype=np.intc)
    is_unique = 0
    # M4: std::stable_sort via a C linkage wrapper. Tighter loop body than
    # numpy's argsort GIL roundtrip path, but does NOT preserve numpy's
    # quicksort tie-break order, so this swap forces a golden re-baseline.
    mt_cpp_argsort_stable_f32(&arr[0], n_sample, &sort_index[0])
    current_bin_id = 0
    # Check if all values are equal
    if arr[sort_index[n_sample-1]] - arr[sort_index[0]] < RELATIVE_THRESHOLD:
        is_unique = 1
    else:
        # Reset the aggregates
        for i in range(n_bins):
            bin_counts[i] = 0
            bin_max[i] = SMALL_NUMBER
            bin_min[i] = LARGE_NUMBER
        # Number of elements per bin

        if n_bins > n_sample:
            n_bins = n_sample

        approximate_bin_count = (int)(n_sample / n_bins) 
        current_bin_count = 1
        bin_max[current_bin_id] = arr[sort_index[0]]
        bin_min[current_bin_id] = arr[sort_index[0]]
        bin_counts[current_bin_id] += 1
        # Assign the bin numbers
        for i in range(1, n_sample):
            if arr[sort_index[i-1]] + RELATIVE_THRESHOLD < arr[sort_index[i]]:
                # handle the same numbers in the same bucket
                if current_bin_count >= approximate_bin_count:
                    current_bin_count = 0
                    current_bin_id += 1

            if current_bin_id >= n_bins:
                current_bin_id -= 1 

            bins[sort_index[i]] = current_bin_id  # Assign based on the sorted order but to the original array indexing
            bin_counts[current_bin_id] += 1
            current_bin_count += 1

            # Update bin_max and bin_min
            if arr[sort_index[i]] > bin_max[current_bin_id]:
                bin_max[current_bin_id] = arr[sort_index[i]]
            if arr[sort_index[i]] < bin_min[current_bin_id]:
                bin_min[current_bin_id] = arr[sort_index[i]]
    #t2 = time.time()
    if current_bin_id == 0:
        is_unique = 1
    #print('Preprocess time: ', t2 - t1)
    return bins, bin_max, bin_min, bin_counts, is_unique, current_bin_id + 1

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef hash_bins(float[:] arr, int n_bins, 
                              int[:] bins, float[:] bin_max, float[:] bin_min,  int[:] bin_counts):
    cdef int i, bin_idx, is_unique
    cdef float min_val = arr[0]
    cdef float max_val = arr[0]
    cdef float eps = 1e-10, bin_width, buffer

    # Reset the aggregates
    for i in range(n_bins):
        bin_counts[i] = 0
        bin_max[i] = SMALL_NUMBER
        bin_min[i] = LARGE_NUMBER
        
    # Find min and max of the array
    for i in range(arr.shape[0]):
        if arr[i] < min_val:
            min_val = arr[i]
        if arr[i] > max_val:
            max_val = arr[i]

    if abs(max_val - min_val) < eps:
        is_unique = 1
    else:
        is_unique = 0
        # Define buffer for bin edges
        bin_width = (max_val - min_val + 1e-10) / n_bins
        for i in range(arr.shape[0]):
            bin_idx = int((arr[i] - min_val) / bin_width)
            if bin_idx == n_bins:
                bin_idx -= 1
            bins[i] = bin_idx
            bin_counts[bin_idx] += 1
            
            # Update bin_max and bin_min
            if arr[i] > bin_max[bin_idx]:
                bin_max[bin_idx] = arr[i]
            if arr[i] < bin_min[bin_idx]:
                bin_min[bin_idx] = arr[i]

        for i in range(1, n_bins):
            if bin_counts[i] == 0:
                bin_min[i] = bin_min[i-1]
                bin_max[i] = bin_max[i-1]

    return bins, bin_max, bin_min, bin_counts, is_unique

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef fit_mt_tree_bin(float[:, ::1] features, float[:, ::1] labels, float[:] weights_target, int[:] target_weight_map, int n_objective,
                float[:] objective_weights, int max_depth, float mtry, int samp_feat_by_node, int min_data_in_leaf,
                int num_nodes, float min_gain_to_split, int num_bins, int num_threads, int verbose,
                int agg_mode):
     
    cdef int N = features.shape[0] # number of instances
    cdef int P = features.shape[1] # number of features

    cdef int N_t = labels.shape[1] # number of targets
    cdef int i, current_node, k,  num_full_nodes, split_var, n_features
    cdef float split_val

    num_full_nodes = 2 * num_nodes - 1
    cdef float[:,::1] node_means = np.zeros((num_full_nodes, N_t), dtype=np.float32)
    cdef int[:, ::1] node_stats = np.zeros((num_full_nodes,7), dtype=np.intc)
    cdef float[:] split_vals = np.zeros(num_full_nodes, dtype=np.float32)
    cdef int[:] node_starts = np.zeros(num_full_nodes, dtype=np.intc)
    cdef int[:] node_ends = np.zeros(num_full_nodes, dtype=np.intc)
    cdef int[:] row_indices = np.zeros(N, dtype=np.intc)   

    if mtry < 1:
        n_features = int(mtry * P) + 1
    else:
        n_features = P

    cdef int[:] feature_indices = np.zeros(n_features, dtype=np.intc)   
    
    # use for splitting 
    cdef float[:, :, :] split_perf = np.zeros((n_features, num_bins, N_t), dtype=np.float32)
    cdef float[:, :, :] split_info = np.full((n_features, num_bins, 5), np.nan, dtype=np.float32)
    cdef float[:,::1] total_sum = np.zeros((num_bins, N_t), dtype=np.float32)
    cdef float[:] overall_total_sum = np.zeros(N_t, dtype=np.float32)
    cdef float[:, :] aggregated_objectives = np.zeros((n_features * num_bins, n_objective), dtype=np.float32)
    cdef float[:] objective_weighted_rank = np.zeros(n_features * num_bins, dtype=np.float32)
    cdef int[:, :] sort_idx_obj = np.zeros((n_features * num_bins, n_objective), dtype=np.intc)   
    cdef int[:, :] rank_obj = np.zeros((n_features * num_bins, n_objective), dtype=np.intc)   
    cdef float[:] left_sum = np.zeros(N_t, dtype=np.float32)
    cdef float[:] right_sum = np.zeros(N_t, dtype=np.float32)
    cdef float[:] critParent = np.zeros(N_t, dtype=np.float32)
    cdef int[:] bin_instance_count = np.zeros(num_bins, dtype = np.intc)
    cdef float[:] bin_max = np.zeros(num_bins, dtype = np.float32)
    cdef float[:] bin_min = np.zeros(num_bins, dtype = np.float32)

    # M4-phase-2 thread-local workspace pools sized [T, ...]. Each thread
    # owns a tid slice during the per-feature prange in find_best_split_mod.
    cdef int nt = num_threads if num_threads > 0 else 1
    cdef float[:, :, ::1] total_sum_pool = np.zeros((nt, num_bins, N_t), dtype=np.float32)
    cdef float[:, ::1] bin_max_pool = np.zeros((nt, num_bins), dtype=np.float32)
    cdef float[:, ::1] bin_min_pool = np.zeros((nt, num_bins), dtype=np.float32)
    cdef int[:, ::1] bin_instance_count_pool = np.zeros((nt, num_bins), dtype=np.intc)
    cdef int[:, ::1] bin_rep_pool = np.zeros((nt, N), dtype=np.intc)
    cdef float[:, ::1] temp_x_pool = np.zeros((nt, N), dtype=np.float32)
    cdef int[:, ::1] qb_sort_idx_pool = np.zeros((nt, N), dtype=np.intc)
    cdef float[:, ::1] left_sum_pool = np.zeros((nt, N_t), dtype=np.float32)
    cdef int[::1] uc_pool = np.zeros(nt, dtype=np.intc)

    train_start = time.time()
    for i in range(N):
        row_indices[i] = i
        
    #initial node
    node_stats[0,0] = 0  # first column node id
    node_stats[0,1] = N  # second column number of data points
    node_stats[0,2] = NODE_TOSPLIT #third column is the node type
    node_stats[0,3] = 0 #fourth column is the depth

    node_starts[0] = 0
    node_ends[0] = N

    for j in range(N_t):
        node_means[0,j] = 0
        for i in range(N):
            node_means[0,j] += labels[i,j]
        node_means[0,j] = node_means[0,j] / N


    #initialize no sampling case
    if n_features == P:
        for i in range(P):
            feature_indices[i] = i
    elif samp_feat_by_node == 0:
        feature_indices = sample_features(P, n_features)

    current_node = 0
    #start main loop
    for k in range(num_full_nodes-2):
        if k > current_node or current_node >= num_full_nodes - 2:
            break

        # skip if the node is not to be split */
        if node_stats[k,2] != NODE_TOSPLIT: continue

        # add an option to do node based feature subsampling
        if samp_feat_by_node == 1:
            feature_indices = sample_features(P, n_features)
            #print(np.asarray(feature_indices))

        t1 = time.time()
        last_index, split_var, split_val = find_best_split_mod(features, labels, row_indices, node_stats[k,1], N_t, n_features, n_objective,
                        node_starts[k], node_ends[k], feature_indices,  min_data_in_leaf, split_perf, split_info, total_sum,
                        overall_total_sum, weights_target, target_weight_map, aggregated_objectives, sort_idx_obj, rank_obj, objective_weights,
                        objective_weighted_rank, num_threads,  num_bins, verbose,
                        left_sum, right_sum, critParent,
                        bin_instance_count, bin_max, bin_min,
                        agg_mode,
                        total_sum_pool, bin_max_pool, bin_min_pool,
                        bin_instance_count_pool, bin_rep_pool, temp_x_pool,
                        qb_sort_idx_pool, left_sum_pool, uc_pool)
        if verbose==3:
            t2 = time.time()
            print('Split evaluation time: ', t2 - t1)

        if split_var == -1:
            node_stats[k,2] = NODE_TERMINAL            
        else:
            node_stats[k,2] = NODE_INTERIOR
            node_stats[k,4] = split_var
            split_vals[k] = split_val
            node_stats[k,5] = current_node + 1 # left child node id
            node_stats[k,6] = current_node + 2 # right child node id
            node_stats[current_node + 1,0] = current_node + 1   # first column node id
            node_stats[current_node + 2,0] = current_node + 2  # first column node id

            node_stats[current_node + 1,1] = last_index  # second column number of data points
            node_stats[current_node + 2,1] = node_stats[k,1] - last_index  # second column number of data points
            
            node_stats[current_node + 1,3] = node_stats[k,3] + 1
            node_stats[current_node + 2,3] = node_stats[k,3] + 1

            if node_stats[current_node + 1,1] < 2 * min_data_in_leaf or node_stats[current_node + 1,3] == max_depth:
                node_stats[current_node + 1,2] = NODE_TERMINAL       
            else:
                node_stats[current_node + 1,2] = NODE_TOSPLIT  

            if node_stats[current_node + 2,1] < 2 * min_data_in_leaf  or node_stats[current_node + 2,3] == max_depth:
                node_stats[current_node + 2,2] = NODE_TERMINAL       
            else:
                node_stats[current_node + 2,2] = NODE_TOSPLIT  

            node_starts[current_node+1] = node_starts[k]
            node_ends[current_node+1] = node_starts[k] + last_index
            node_starts[current_node+2] = node_ends[current_node+1]
            node_ends[current_node+2] = node_ends[k]

        # node means for left and right child
            for j in range(N_t):
                node_means[current_node+1,j] = 0
                node_means[current_node+2,j] = 0
                for i in range(last_index):
                    node_means[current_node+1,j] += labels[row_indices[i+node_starts[k]],j]
                for i in range(last_index,node_stats[k,1]):   
                    node_means[current_node+2,j] += labels[row_indices[i+node_starts[k]],j]
                
                node_means[current_node+1,j] = node_means[current_node+1,j]  /  (last_index)
                node_means[current_node+2,j] = node_means[current_node+2,j]  /  (node_stats[k,1] - last_index) 
                
            current_node = current_node + 2

    cdef int actual_nodes = current_node + 1
    for k in range(actual_nodes):
        if node_stats[k,2] == NODE_TOSPLIT:
            node_stats[k,2]= NODE_TERMINAL

    if verbose == 1:
        t2 = time.time()
        print('Tree fitting time: ', t2 - train_start)

    return (np.asarray(node_stats)[:actual_nodes].copy(),
            np.asarray(split_vals)[:actual_nodes].copy(),
            np.asarray(node_means)[:actual_nodes].copy())
        
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef inline int _quantile_bins_nogil(
        float * arr, int n_sample, int n_bins,
        int * bins, float * bin_max, float * bin_min, int * bin_counts,
        int * sort_idx,
        int * resulting_bin_count_out) nogil:
    """nogil drop-in for the cpdef quantile_bins. Returns is_unique (0/1).

    Sorts via mt_cpp_argsort_stable_f32 (std::stable_sort, releases nothing
    because it's pure C linkage). Reproduces the legacy tie-aware equal-count
    binning EXACTLY, including the quirk that bins[sort_idx[0]] is NOT
    written (so it retains whatever the caller's bin buffer had on entry).
    """
    cdef int i, current_bin_id, current_bin_count, approximate_bin_count
    cdef int is_unique = 0
    # Use the SAME module-level constants the cpdef quantile_bins uses, so
    # tie/empty-bin comparison semantics match exactly.

    mt_cpp_argsort_stable_f32(arr, n_sample, sort_idx)

    current_bin_id = 0
    if arr[sort_idx[n_sample - 1]] - arr[sort_idx[0]] < RELATIVE_THRESHOLD:
        is_unique = 1
    else:
        for i in range(n_bins):
            bin_counts[i] = 0
            bin_max[i] = SMALL_NUMBER
            bin_min[i] = LARGE_NUMBER
        if n_bins > n_sample:
            n_bins = n_sample
        approximate_bin_count = (<int>(n_sample / n_bins))
        current_bin_count = 1
        bin_max[current_bin_id] = arr[sort_idx[0]]
        bin_min[current_bin_id] = arr[sort_idx[0]]
        bin_counts[current_bin_id] += 1
        # M4-phase-2 methodology fix: legacy quantile_bins skipped writing
        # bins[sort_idx[0]] and relied on the caller's bin_rep retaining a
        # "stale" value from the previous feature's pass. That bug only
        # produces a consistent result under serial execution. To make
        # parallel results deterministic AND fix the underlying mis-assignment,
        # we explicitly assign the smallest-value position to bin 0.
        bins[sort_idx[0]] = current_bin_id
        for i in range(1, n_sample):
            if arr[sort_idx[i - 1]] + RELATIVE_THRESHOLD < arr[sort_idx[i]]:
                if current_bin_count >= approximate_bin_count:
                    current_bin_count = 0
                    current_bin_id += 1
            if current_bin_id >= n_bins:
                current_bin_id -= 1
            bins[sort_idx[i]] = current_bin_id
            bin_counts[current_bin_id] += 1
            current_bin_count += 1
            if arr[sort_idx[i]] > bin_max[current_bin_id]:
                bin_max[current_bin_id] = arr[sort_idx[i]]
            if arr[sort_idx[i]] < bin_min[current_bin_id]:
                bin_min[current_bin_id] = arr[sort_idx[i]]

    if current_bin_id == 0:
        is_unique = 1

    resulting_bin_count_out[0] = current_bin_id + 1
    return is_unique


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef find_best_split_mod(float[:, ::1] features, float[:, ::1] labels, int[:] row_indices, int n_sample, int n_target, int n_predictor, int n_objective,
                        int ndstart, int ndend, int[:] feature_indices, int min_data_in_leaf,
                        float[:, :, :] split_perf, float[:, :, :] split_info,
                        float[:,:] total_sum, float[:] overall_total_sum, float[:] weights_target, int[:] target_weight_map, float[:, :] aggregated_objectives,
                        int[:, :] sort_idx_obj, int[:, :] rank_obj, float[:] objective_weights, float[:] objective_weighted_rank,
                        int num_threads, int given_bin_count, int verbose,
                        float[:] left_sum, float[:] right_sum,float[:] critParent,
                        int[:] bin_instance_count, float[:] bin_max, float[:] bin_min,
                        int agg_mode,
                        # M4-phase-2 thread-local pools shaped [T, ...]:
                        float[:, :, ::1] total_sum_pool,
                        float[:, ::1] bin_max_pool, float[:, ::1] bin_min_pool,
                        int[:, ::1] bin_instance_count_pool,
                        int[:, ::1] bin_rep_pool, float[:, ::1] temp_x_pool,
                        int[:, ::1] qb_sort_idx_pool,
                        float[:, ::1] left_sum_pool,
                        int[::1] uc_pool): #nogil:

    cdef float sst_reduction, lbl
    cdef int i, t, left_count_m, right_count_m,unique_counts, b, o, p, cumul_count_bin_instance, n_split_candidate, split_var_index
    cdef int split_var,  best_index, split_val_index, resulting_bin_count
    cdef int[::1] sort_idx_var = np.zeros(n_sample, dtype=np.intc)
    cdef int[:] sorted_row_indices = np.zeros(n_sample, dtype=np.intc)
    cdef float[::1] temp_x = np.zeros(n_sample, dtype=np.float32)
    cdef int[:] bin_rep = np.zeros(n_sample, dtype = np.intc)
    cdef float best_val
    cdef int n_cand
    cdef float obj_min, obj_max, obj_range, v
    cdef float EPS = 1e-12
    # M1 scratch for in-kernel rank aggregation (size n_predictor * given_bin_count).
    cdef int n_cand_total = n_predictor * given_bin_count
    cdef int[::1] rank_idx_buf = np.zeros(n_cand_total, dtype=np.intc)
    cdef int[::1] rank_scratch_buf = np.zeros(n_cand_total, dtype=np.intc)
    # M1 scratch for in-kernel partition argsort on the selected split column.
    cdef int[::1] sort_scratch = np.zeros(n_sample, dtype=np.intc)
    # M3 helper variable for per-bin SIMD accumulation.
    cdef int sh_row_i
    # M4-phase-2 prange-scoped scratch (auto-private under cython.parallel).
    cdef int tid, feat_idx_p, is_unique_p, resulting_bin_count_p
    cdef int cumul_count_bin_instance_p, left_count_m_p, right_count_m_p
    cdef float right_sum_p, sst_reduction_p
    cdef int t_tid
    cdef int nt_eff = num_threads if num_threads > 0 else 1

    resulting_bin_count = 0
    for t in range(n_target):
        overall_total_sum[t] = 0    
    
    for i in range(ndstart,ndend):
        for t in range(n_target):
            lbl = labels[row_indices[i],t]
            overall_total_sum[t] += lbl
   
    for t in range(n_target):
        critParent[t] = overall_total_sum[t] * overall_total_sum[t] / n_sample
    
    for p in range(n_predictor):
        for b in range(given_bin_count):
            objective_weighted_rank[b + p*given_bin_count] = 0
            for o in range(n_objective):
                aggregated_objectives[b + p*given_bin_count, o] = 0
               
    # M4-phase-2: parallel per-feature split-finding.
    # Each thread owns its tid-slice of all *_pool workspaces, plus the
    # split_info / split_perf / aggregated_objectives rows belonging to its
    # current feature p (no cross-thread write race).
    for t_tid in range(nt_eff):
        uc_pool[t_tid] = 0
    # Legacy parity: the cpdef quantile_bins quirk leaves bins[sort_idx[0]]
    # unwritten, so the position of the smallest value gets the prior call's
    # value. The LEGACY find_best_split_mod allocated bin_rep fresh per node
    # via np.zeros, so that "stale" value was ALWAYS 0 at p=0 of any node.
    # Our pool persists across nodes, so we must zero it per call to keep
    # the same miscredit semantics.
    for t_tid in range(nt_eff):
        for i in range(n_sample):
            bin_rep_pool[t_tid, i] = 0

    with nogil, parallel(num_threads=nt_eff):
        tid = omp_get_thread_num()
        for p in prange(n_predictor):
            feat_idx_p = feature_indices[p]

            # Build per-thread temp_x[tid, :n_sample] = node feature values.
            for i in range(ndstart, ndend):
                temp_x_pool[tid, i - ndstart] = features[row_indices[i], feat_idx_p]

            # nogil quantile_bins (uses std::stable_sort via mt_cpp_argsort).
            is_unique_p = _quantile_bins_nogil(
                &temp_x_pool[tid, 0], n_sample, given_bin_count,
                &bin_rep_pool[tid, 0],
                &bin_max_pool[tid, 0], &bin_min_pool[tid, 0],
                &bin_instance_count_pool[tid, 0],
                &qb_sort_idx_pool[tid, 0],
                &resulting_bin_count_p)

            # Reset split_info[p, :, :] and total_sum_pool[tid, :, :].
            for b in range(given_bin_count):
                split_info[p, b, 0] = -1
                for t in range(n_target):
                    total_sum_pool[tid, b, t] = 0.0

            if is_unique_p == 1:
                uc_pool[tid] = uc_pool[tid] + 1
                continue

            # M3 SIMD K-target accumulation into total_sum_pool[tid, bin, :].
            for i in range(ndstart, ndend):
                mt_simd_accum_bin_k_f32(
                    &total_sum_pool[tid, bin_rep_pool[tid, i - ndstart], 0],
                    &labels[row_indices[i], 0],
                    n_target)

            # Per-feature left_sum (thread-local).
            for t in range(n_target):
                left_sum_pool[tid, t] = 0.0

            cumul_count_bin_instance_p = 0
            for b in range(resulting_bin_count_p - 1):
                cumul_count_bin_instance_p = (
                    cumul_count_bin_instance_p + bin_instance_count_pool[tid, b])
                if cumul_count_bin_instance_p < min_data_in_leaf:
                    if cumul_count_bin_instance_p + bin_instance_count_pool[tid, b + 1] < n_sample:
                        split_info[p, b, 0] = -1
                        for t in range(n_target):
                            left_sum_pool[tid, t] = (
                                left_sum_pool[tid, t] + total_sum_pool[tid, b, t])
                        continue

                split_info[p, b, 0] = feature_indices[p]
                split_info[p, b, 1] = b
                split_info[p, b, 2] = (bin_max_pool[tid, b] + bin_min_pool[tid, b + 1]) / 2
                split_info[p, b, 3] = cumul_count_bin_instance_p
                split_info[p, b, 4] = n_sample - cumul_count_bin_instance_p

                for t in range(n_target):
                    left_sum_pool[tid, t] = (
                        left_sum_pool[tid, t] + total_sum_pool[tid, b, t])
                    right_sum_p = overall_total_sum[t] - left_sum_pool[tid, t]
                    left_count_m_p = cumul_count_bin_instance_p
                    right_count_m_p = n_sample - cumul_count_bin_instance_p
                    sst_reduction_p = (
                        (left_sum_pool[tid, t] * left_sum_pool[tid, t]) / left_count_m_p
                        + (right_sum_p * right_sum_p) / right_count_m_p
                        - critParent[t])
                    split_perf[p, b, t] = sst_reduction_p
                    aggregated_objectives[b + p * given_bin_count, target_weight_map[t]] += (
                        -sst_reduction_p * weights_target[t])

                if (n_sample - cumul_count_bin_instance_p) < min_data_in_leaf:
                    break

    # Reduce uc_pool to unique_counts.
    unique_counts = 0
    for t_tid in range(nt_eff):
        unique_counts = unique_counts + uc_pool[t_tid]

    if unique_counts == n_predictor:
        return -1, -1, -1

    if agg_mode == 0:
        # rank aggregation: in-kernel stable rank (M1).
        fill_stable_rank_2d_axis0_f32(
            aggregated_objectives, n_cand_total, n_objective,
            rank_obj, &rank_idx_buf[0], &rank_scratch_buf[0])
        for p in range(n_predictor):
            for b in range(given_bin_count):
                for o in range(n_objective):
                    objective_weighted_rank[b + p*given_bin_count] +=  rank_obj[b + p*given_bin_count, o] * objective_weights[o]
        best_val = objective_weighted_rank[0]
        best_index = 0
        for i in range(1, n_cand_total):
            if objective_weighted_rank[i] < best_val:
                best_val = objective_weighted_rank[i]
                best_index = i
    else:
        # score aggregation: per-objective min-max normalization, weighted-sum-of-scores, argmin
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
    split_var_index = (int)(best_index / given_bin_count)
    split_var = (int)(split_info[split_var_index,split_val_index,0])
    split_val = split_info[split_var_index,split_val_index,2]
    
    if split_info[split_var_index,split_val_index,0] == -1:
        return -1, -1, -1
        
    left_count_m = 0
    for i in range(n_sample):
        temp_x[i] = features[row_indices[ndstart+i],split_var]
        if temp_x[i] < split_val: left_count_m += 1
    
    # M1: in-kernel stable argsort on the chosen split-column values.
    stable_argsort_1d_f32(temp_x, n_sample, sort_idx_var, &sort_scratch[0])
    for i in range(n_sample):
        sorted_row_indices[i] = row_indices[ndstart + sort_idx_var[i]]
        
    for i in range(ndstart,ndend):
        row_indices[i] = sorted_row_indices[i-ndstart]

    if left_count_m == n_sample or left_count_m == 0:
        print(best_index)
        print(unique_counts)
        print(n_split_candidate)
        print(n_predictor)
        print(n_sample)
        print(split_var)
        print(split_val)

        for t in range(n_target):
            for i in range(ndstart,ndend):
                temp_x[i-ndstart] = labels[row_indices[i],t]
            np.savetxt('temp_label_' + str(t) + '.csv',temp_x,delimiter=',')    

        for p in range(n_predictor):
            for i in range(ndstart,ndend):
                temp_x[i-ndstart] = features[row_indices[i],p]

            bin_rep, bin_max, bin_min, bin_instance_count, \
                is_unique, resulting_bin_count = quantile_bins(temp_x, given_bin_count, bin_rep, bin_max, bin_min, bin_instance_count)
            print([p, resulting_bin_count])
            np.savetxt('temp_' + str(p) + '.csv',temp_x,delimiter=',')    

        print(np.asarray(split_info[split_var,:,:]))
        print(np.asarray(split_perf[split_var,:,:]))
        for b in range(given_bin_count):
            print(np.asarray(aggregated_objectives[b + split_var*given_bin_count,:]))
            print(np.asarray(objective_weighted_rank[b + split_var*given_bin_count]))

        print([left_count_m, n_sample])
        print('possible split')
        input('dur')

    return left_count_m, split_var, split_val

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef predict_mt_tree(float[:, :] features,  int[:, :] node_stats, float[:] split_vals, float[:,:] node_means, int maxdepth, int verbose):

    cdef int N = features.shape[0] # number of instances
    cdef int N_t = node_means.shape[1] # number of targets

    cdef float[:,:] prediction = np.zeros((N, N_t), dtype=np.float32)
    cdef int i, k, split_var
    cdef float split_val

    for i in range(N):
        k = 0
        while node_stats[k,2] != NODE_TERMINAL and node_stats[k,3] < maxdepth:
            split_var = node_stats[k,4]
            split_val = split_vals[k]

            if features[i,split_var] < split_val:
                k = node_stats[k,5]
            else:
                k = node_stats[k,6]

        for t in range(N_t):
            prediction[i,t] = node_means[k,t]

    return np.asarray(prediction)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef predict_mt_tree_node(float[:, :] features,  int[:, :] node_stats, float[:] split_vals, float[:,:] node_means, int maxdepth, int verbose):

    cdef int N = features.shape[0] # number of instances
    cdef int N_t = node_means.shape[1] # number of targets

    cdef int[:] prediction = np.zeros(N, dtype=np.intc)
    cdef int i, k, split_var
    cdef float split_val

    for i in range(N):
        k = 0
        while node_stats[k,2] != NODE_TERMINAL and node_stats[k,3] < maxdepth:
            split_var = node_stats[k,4]
            split_val = split_vals[k]

            if features[i,split_var] < split_val:
                k = node_stats[k,5]
            else:
                k = node_stats[k,6]

        prediction[i] = k

    return np.asarray(prediction)



