#include "sort_cpp.h"

#include <algorithm>

extern "C" void mt_cpp_argsort_stable_f32(const float * arr, int n, int * out_idx)
{
    for (int i = 0; i < n; ++i) out_idx[i] = i;
    std::stable_sort(out_idx, out_idx + n,
                     [arr](int a, int b) { return arr[a] < arr[b]; });
}
