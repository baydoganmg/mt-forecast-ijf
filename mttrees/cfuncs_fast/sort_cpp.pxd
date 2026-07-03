cdef extern from "sort_cpp.h" nogil:
    void mt_cpp_argsort_stable_f32(const float * arr, int n, int * out_idx)
