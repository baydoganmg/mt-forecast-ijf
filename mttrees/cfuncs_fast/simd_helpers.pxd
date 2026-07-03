cdef extern from "simd_helpers.h" nogil:
    void mt_simd_accum_bin_k_f32(float * dst, const float * src, int n_target)
    void mt_simd_add_k_to_neg_k_f32(float * dst, const float * src, int n_target)
