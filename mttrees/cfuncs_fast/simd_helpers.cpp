#include "simd_helpers.h"

#if defined(__AVX2__)
  #include <immintrin.h>
#endif

/* dst[t] += src[t] for t in [0, n_target). 8-wide AVX2 main loop + scalar tail. */
void mt_simd_accum_bin_k_f32(
        float * __restrict__ dst,
        const float * __restrict__ src,
        int n_target)
{
    int t = 0;
#if defined(__AVX2__)
    for (; t + 8 <= n_target; t += 8) {
        __m256 a = _mm256_loadu_ps(dst + t);
        __m256 b = _mm256_loadu_ps(src + t);
        _mm256_storeu_ps(dst + t, _mm256_add_ps(a, b));
    }
#endif
    for (; t < n_target; ++t) {
        dst[t] += src[t];
    }
}

/* Unused reserve slot (kept for future M3 expansion). */
void mt_simd_add_k_to_neg_k_f32(
        float * __restrict__ dst,
        const float * __restrict__ src,
        int n_target)
{
    int t = 0;
#if defined(__AVX2__)
    for (; t + 8 <= n_target; t += 8) {
        __m256 a = _mm256_loadu_ps(dst + t);
        __m256 b = _mm256_loadu_ps(src + t);
        _mm256_storeu_ps(dst + t, _mm256_sub_ps(a, b));
    }
#endif
    for (; t < n_target; ++t) {
        dst[t] -= src[t];
    }
}
