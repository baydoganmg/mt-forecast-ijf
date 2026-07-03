#pragma once

#include <cstdint>

namespace mttrees::cpp_kernel {

constexpr float RELATIVE_THRESHOLD = 1e-7f;

/* sprint-6: order-preserving fp32 -> uint32 key transform for the
 * packed-key argsort. Strictly monotone over all non-NaN floats:
 *   f1 < f2  <=>  key(f1) < key(f2)
 * -0.0 is canonicalized to +0.0 BEFORE the transform so float-equal
 * values (incl. +/-0.0) map to EQUAL keys; combined with the row index
 * packed in the low 32 bits of the sort key, equal floats are ordered
 * by original index — byte-identical tie-breaking to std::stable_sort
 * with comparator `arr[a] < arr[b]`. NaN inputs are not supported
 * (same precondition as the previous stable_sort comparator, whose
 * behavior with NaNs was already unspecified). */
inline uint32_t fp32_order_key(float f) noexcept {
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    if (u == 0x80000000u) u = 0;                       // -0.0 -> +0.0
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u); // monotone flip
}

/* Stable argsort of [0, n_sample) by arr value via packed 64-bit keys:
 * key = (fp32_order_key(arr[i]) << 32) | i, sorted with std::sort.
 * Produces the IDENTICAL permutation to
 *   std::stable_sort(idx, idx+n, [arr](a,b){ return arr[a] < arr[b]; })
 * for NaN-free input (see fp32_order_key proof above), but with no
 * indirect comparator loads. key_scratch: caller scratch, length
 * n_sample. Results land in sort_idx. */
void packed_key_argsort_f32(
    const float * arr,
    int n_sample,
    uint64_t * key_scratch,
    int * sort_idx
) noexcept;

/* nogil-equivalent quantile binning. Mirrors the cfuncs_fast
 * _quantile_bins_nogil semantics, including the M4-phase-2 methodology fix
 * (explicit write at bins[sort_idx[0]] = 0). sprint-6: argsort via
 * packed_key_argsort_f32 (bit-identical permutation to the previous
 * std::stable_sort, measurably faster).
 *
 * Parameters:
 *   arr           input feature values for this node, length n_sample
 *   n_sample      number of samples in the current node
 *   n_bins        target number of bins (may be clamped to n_sample)
 *   bins          output bin id per sample, length n_sample
 *   bin_max       output max value per bin, length n_bins
 *   bin_min       output min value per bin, length n_bins
 *   bin_counts    output count per bin, length n_bins
 *   sort_idx      scratch buffer for the argsort, length n_sample
 *   key_scratch   scratch buffer for packed keys, length n_sample
 *
 * Returns:
 *   pair<is_unique, resulting_bin_count>
 *     is_unique = 1 if all values equal OR only one bin filled
 *     resulting_bin_count = number of bins actually used
 */
struct QuantileBinsResult {
    int is_unique;
    int resulting_bin_count;
};

QuantileBinsResult quantile_bins_f32(
    const float * arr,
    int n_sample,
    int n_bins,
    int * bins,
    float * bin_max,
    float * bin_min,
    int * bin_counts,
    int * sort_idx,
    uint64_t * key_scratch
) noexcept;

}  // namespace mttrees::cpp_kernel
