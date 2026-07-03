#ifndef MTTREES_SIMD_HELPERS_H
#define MTTREES_SIMD_HELPERS_H

/* M3: per-bin K-target accumulation primitives.
 *
 * Hot loop in the prebin and selfhash kernels: for each (row, feature) pair
 * the row's K-vector labels[row, :] is added to a per-bin accumulator
 * total_sum[bin, :]. K is small and fixed per fit (typically 12, 17, or 25).
 *
 * The auto-vectorizer already does some of this under -O3 -ftree-vectorize,
 * but the memoryview indexing surrounding it leaves loop-invariant address
 * computations the optimizer can't always hoist. Exposing the inner add as
 * a plain pointer-pointer call lets us hand-vectorize K=8 chunks with AVX2
 * FMA-friendly adds and fall back to scalar for the tail.
 */

#ifdef __cplusplus
extern "C" {
#endif

void mt_simd_accum_bin_k_f32(
        float * __restrict__ dst,
        const float * __restrict__ src,
        int n_target);

void mt_simd_add_k_to_neg_k_f32(
        float * __restrict__ dst,
        const float * __restrict__ src,
        int n_target);

#ifdef __cplusplus
}
#endif

#endif /* MTTREES_SIMD_HELPERS_H */
