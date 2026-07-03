# distutils: language = c++
# cython: language_level=3
import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import parallel, prange
from cython import boundscheck, wraparound
from libc.math cimport abs
import time

cdef int NODE_TERMINAL = -1
cdef int NODE_TOSPLIT = -2
cdef int NODE_INTERIOR = -3
cdef double RELATIVE_THRESHOLD = 0.0000001
cdef double SMALL_NUMBER = -float('inf')
cdef double LARGE_NUMBER = float('inf')

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
cpdef quantile_bins(double[:] arr, int n_bins, 
                              int[:] bins, double[:] bin_max, double[:] bin_min,  int[:] bin_counts):
    cdef int i, n_sample, is_unique, approximate_bin_count, current_bin_id, current_bin_count
    n_sample = arr.shape[0]
    cdef int[:] sort_index = np.zeros(n_sample, dtype=np.intc)
    is_unique = 0
    #t1 = time.time()
    sort_index = np.argsort(np.asarray(arr)).astype(np.intc)
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
cpdef hash_bins(double[:] arr, int n_bins, 
                              int[:] bins, double[:] bin_max, double[:] bin_min,  int[:] bin_counts):
    cdef int i, bin_idx, is_unique
    cdef double min_val = arr[0]
    cdef double max_val = arr[0]
    cdef double eps = 1e-10, bin_width, buffer

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
cpdef fit_mt_tree_bin(double[:, ::1] features, double[:, ::1] labels, double[:] weights_target, int[:] target_weight_map, int n_objective,
                double[:] objective_weights, int max_depth, double mtry, int samp_feat_by_node, int min_data_in_leaf,
                int num_nodes, double min_gain_to_split, int num_bins, int num_threads, int verbose,
                int agg_mode):
     
    cdef int N = features.shape[0] # number of instances
    cdef int P = features.shape[1] # number of features

    cdef int N_t = labels.shape[1] # number of targets
    cdef int i, current_node, k,  num_full_nodes, split_var, n_features
    cdef double split_val

    num_full_nodes = 2 * num_nodes - 1
    cdef double[:,::1] node_means = np.zeros((num_full_nodes, N_t), dtype=np.double)
    cdef int[:, ::1] node_stats = np.zeros((num_full_nodes,7), dtype=np.intc)
    cdef double[:] split_vals = np.zeros(num_full_nodes, dtype=np.double)
    cdef int[:] node_starts = np.zeros(num_full_nodes, dtype=np.intc)
    cdef int[:] node_ends = np.zeros(num_full_nodes, dtype=np.intc)
    cdef int[:] row_indices = np.zeros(N, dtype=np.intc)   

    if mtry < 1:
        n_features = int(mtry * P) + 1
    else:
        n_features = P

    cdef int[:] feature_indices = np.zeros(n_features, dtype=np.intc)   
    
    # use for splitting 
    cdef double[:, :, :] split_perf = np.zeros((n_features, num_bins, N_t), dtype=np.double)
    cdef double[:, :, :] split_info = np.full((n_features, num_bins, 5), np.nan, dtype=np.double)
    cdef double[:,::1] total_sum = np.zeros((num_bins, N_t), dtype=np.double)
    cdef double[:] overall_total_sum = np.zeros(N_t, dtype=np.double)
    cdef double[:, :] aggregated_objectives = np.zeros((n_features * num_bins, n_objective), dtype=np.double)
    cdef double[:] objective_weighted_rank = np.zeros(n_features * num_bins, dtype=np.double)
    cdef int[:, :] sort_idx_obj = np.zeros((n_features * num_bins, n_objective), dtype=np.intc)   
    cdef int[:, :] rank_obj = np.zeros((n_features * num_bins, n_objective), dtype=np.intc)   
    cdef double[:] left_sum = np.zeros(N_t, dtype=np.double)
    cdef double[:] right_sum = np.zeros(N_t, dtype=np.double)
    cdef double[:] critParent = np.zeros(N_t, dtype=np.double)
    cdef int[:] bin_instance_count = np.zeros(num_bins, dtype = np.intc)
    cdef double[:] bin_max = np.zeros(num_bins, dtype = np.double)
    cdef double[:] bin_min = np.zeros(num_bins, dtype = np.double)   

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
                        agg_mode)
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
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef find_best_split_mod(double[:, ::1] features, double[:, ::1] labels, int[:] row_indices, int n_sample, int n_target, int n_predictor, int n_objective,
                        int ndstart, int ndend, int[:] feature_indices, int min_data_in_leaf,
                        double[:, :, :] split_perf, double[:, :, :] split_info,
                        double[:,:] total_sum, double[:] overall_total_sum, double[:] weights_target, int[:] target_weight_map, double[:, :] aggregated_objectives,
                        int[:, :] sort_idx_obj, int[:, :] rank_obj, double[:] objective_weights, double[:] objective_weighted_rank,
                        int num_threads, int given_bin_count, int verbose,
                        double[:] left_sum, double[:] right_sum,double[:] critParent,
                        int[:] bin_instance_count, double[:] bin_max, double[:] bin_min,
                        int agg_mode): #nogil:

    cdef double sst_reduction, lbl
    cdef int i, t, left_count_m, right_count_m,unique_counts, b, o, p, cumul_count_bin_instance, n_split_candidate, split_var_index
    cdef int split_var,  best_index, split_val_index, resulting_bin_count
    cdef int[:] sort_idx_var = np.zeros(n_sample, dtype=np.intc)
    cdef int[:] sorted_row_indices = np.zeros(n_sample, dtype=np.intc)
    cdef double[:] temp_x = np.zeros(n_sample, dtype=np.double)
    cdef int[:] bin_rep = np.zeros(n_sample, dtype = np.intc)
    cdef double best_val
    cdef int n_cand
    cdef double obj_min, obj_max, obj_range, v
    cdef double EPS = 1e-12

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
               
    unique_counts = 0
    n_split_candidate = 0
    for p in range(n_predictor):
        #t1 = time.time() 
        for i in range(ndstart,ndend):
            temp_x[i-ndstart] = features[row_indices[i],feature_indices[p]]

        bin_rep, bin_max, bin_min, bin_instance_count, \
            is_unique, resulting_bin_count = quantile_bins(temp_x, given_bin_count, bin_rep, bin_max, bin_min, bin_instance_count)

        for b in range(given_bin_count):
            split_info[p, b, 0] = -1
            for t in range(n_target):
                total_sum[b,t] = 0.0

        if is_unique == 1:
            unique_counts += 1
            continue

        for i in range(ndstart,ndend):
            for t in range(n_target):
                lbl = labels[row_indices[i], t]
                total_sum[bin_rep[i-ndstart],t] += lbl

        for t in range(n_target):
            left_sum[t] = 0

        cumul_count_bin_instance = 0

        for b in range(resulting_bin_count-1):
            cumul_count_bin_instance += bin_instance_count[b]
            if cumul_count_bin_instance < min_data_in_leaf:
                if cumul_count_bin_instance +  bin_instance_count[b+1] < n_sample:
                    split_info[p, b, 0] = -1
                    for t in range(n_target):
                        left_sum[t] += total_sum[b,t] 
                    continue

            n_split_candidate += 1
            split_info[p, b, 0] = feature_indices[p]
            split_info[p, b, 1] = b
            split_info[p, b, 2] = (bin_max[b] + bin_min[b + 1]) / 2
            split_info[p, b, 3] = cumul_count_bin_instance 
            split_info[p, b, 4] = n_sample - cumul_count_bin_instance

            for t in range(n_target):
                left_sum[t] += total_sum[b,t] 
                right_sum[t] = overall_total_sum[t] - left_sum[t] 
                left_count_m = int(split_info[p, b, 3])
                right_count_m = int(split_info[p, b, 4])
                sst_reduction = (left_sum[t] * left_sum[t]) / left_count_m + (right_sum[t] * right_sum[t]) / right_count_m - critParent[t]
                split_perf[p, b, t] = sst_reduction
                aggregated_objectives[b + p*given_bin_count, target_weight_map[t]] += -sst_reduction * weights_target[t]

            if (n_sample - cumul_count_bin_instance) < min_data_in_leaf:
                break

    if unique_counts == n_predictor:
        return -1, -1, -1

    if agg_mode == 0:
        # rank aggregation: double argsort, weighted-sum-of-ranks
        sort_idx_obj = np.argsort(np.asarray(aggregated_objectives), kind='stable',axis=0).astype(np.intc)
        rank_obj = np.argsort(np.asarray(sort_idx_obj), kind='stable',axis=0).astype(np.intc)
        for p in range(n_predictor):
            for b in range(given_bin_count):
                for o in range(n_objective):
                    objective_weighted_rank[b + p*given_bin_count] +=  rank_obj[b + p*given_bin_count, o] * objective_weights[o]
        best_index = np.argmin(np.asarray(objective_weighted_rank)).astype(np.intc)
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
    
    sort_idx_var = np.argsort(np.asarray(temp_x),kind='stable').astype(np.intc)       
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
cpdef predict_mt_tree(double[:, :] features,  int[:, :] node_stats, double[:] split_vals, double[:,:] node_means, int maxdepth, int verbose):

    cdef int N = features.shape[0] # number of instances
    cdef int N_t = node_means.shape[1] # number of targets

    cdef double[:,:] prediction = np.zeros((N, N_t), dtype=np.double)
    cdef int i, k, split_var
    cdef double split_val

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
cpdef predict_mt_tree_node(double[:, :] features,  int[:, :] node_stats, double[:] split_vals, double[:,:] node_means, int maxdepth, int verbose):

    cdef int N = features.shape[0] # number of instances
    cdef int N_t = node_means.shape[1] # number of targets

    cdef int[:] prediction = np.zeros(N, dtype=np.intc)
    cdef int i, k, split_var
    cdef double split_val

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



