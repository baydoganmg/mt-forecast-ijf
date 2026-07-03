#ifndef MTTREES_SORT_CPP_H
#define MTTREES_SORT_CPP_H

/* C linkage wrappers around C++ std::stable_sort for per-feature value sorts.
 *
 * Used inside the selfhash kernel's quantile_bins to replace the legacy
 * np.argsort call. Compared to the hand-written Cython merge sort in
 * sort_helpers.pxd, std::stable_sort uses a tuned introsort+merge hybrid
 * with branchless leaf paths and is typically within 10-30% of numpy's
 * argsort speed on small fp32 columns.
 *
 * NOT a drop-in tie-break match for numpy's argsort (different stable
 * sort), so callers must re-baseline goldens after switching.
 */

#ifdef __cplusplus
extern "C" {
#endif

void mt_cpp_argsort_stable_f32(const float * arr, int n, int * out_idx);

#ifdef __cplusplus
}
#endif

#endif /* MTTREES_SORT_CPP_H */
