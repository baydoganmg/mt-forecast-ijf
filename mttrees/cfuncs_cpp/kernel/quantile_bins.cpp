#include "quantile_bins.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace mttrees::cpp_kernel {

// Below this n_sample, the comparison sort beats radix: radix's fixed
// per-pass overhead (4 x 256-bucket histogram build+clear) dominates for
// small nodes, of which a deep tree has many. Above it, radix's O(n) wins
// over std::sort's O(n log n) -- the upper/large nodes that dominate total
// sort work. Both branches yield the IDENTICAL permutation (see below).
static constexpr int RADIX_MIN_N = 256;

void packed_key_argsort_f32(
    const float * arr,
    int n_sample,
    uint64_t * key_scratch,
    int * sort_idx
) noexcept {
    if (n_sample < RADIX_MIN_N) {
        // Small-n: packed-key comparison sort. Order = (fp32 order key asc,
        // then original index asc) since the index sits in the low 32 bits.
        for (int i = 0; i < n_sample; ++i) {
            key_scratch[i] =
                (uint64_t(fp32_order_key(arr[i])) << 32) | uint32_t(i);
        }
        std::sort(key_scratch, key_scratch + n_sample);
        for (int i = 0; i < n_sample; ++i) {
            sort_idx[i] = int(uint32_t(key_scratch[i]));
        }
        return;
    }

    // Large-n: 4-pass stable LSD radix on the 32-bit order-preserving fp32
    // key, carrying the original index. BIT-IDENTICAL permutation to the
    // comparison branch: LSD radix is stable, so equal keys keep their input
    // order (index ascending), and the passes sort by the key ascending --
    // exactly the (key asc, index asc) order the packed std::sort produces.
    //
    // key_scratch is uint64[n] = 2*n uint32 slots: use as two key buffers.
    // sort_idx and a thread_local buffer are the two index buffers. Both
    // index buffers are written every pass, so neither is read uninitialized.
    uint32_t * ka = reinterpret_cast<uint32_t *>(key_scratch);  // [0, n)
    uint32_t * kb = ka + n_sample;                              // [n, 2n)
    thread_local std::vector<int> idx_scratch;
    idx_scratch.resize(n_sample);
    int * ia = sort_idx;
    int * ib = idx_scratch.data();

    for (int i = 0; i < n_sample; ++i) {
        ka[i] = fp32_order_key(arr[i]);
        ia[i] = i;
    }
    for (int shift = 0; shift < 32; shift += 8) {
        int count[256];
        for (int b = 0; b < 256; ++b) count[b] = 0;
        for (int i = 0; i < n_sample; ++i) ++count[(ka[i] >> shift) & 0xFF];
        int sum = 0;
        for (int b = 0; b < 256; ++b) { int c = count[b]; count[b] = sum; sum += c; }
        for (int i = 0; i < n_sample; ++i) {
            const uint32_t k = ka[i];
            const int pos = count[(k >> shift) & 0xFF]++;
            kb[pos] = k;
            ib[pos] = ia[i];
        }
        std::swap(ka, kb);
        std::swap(ia, ib);
    }
    // 4 passes (even) -> ia is back to sort_idx and holds the final order.
}

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
) noexcept {
    int is_unique = 0;
    int current_bin_id = 0;

    // Stable argsort of indices [0, n_sample) by arr value. sprint-6:
    // packed-key sort — same permutation as the previous stable_sort
    // (proof at fp32_order_key in the header), no indirect comparator.
    packed_key_argsort_f32(arr, n_sample, key_scratch, sort_idx);

    // All-equal short-circuit.
    if (arr[sort_idx[n_sample - 1]] - arr[sort_idx[0]] < RELATIVE_THRESHOLD) {
        return {1, current_bin_id + 1};
    }

    // Reset bin aggregates.
    for (int i = 0; i < n_bins; ++i) {
        bin_counts[i] = 0;
        bin_max[i] = -std::numeric_limits<float>::infinity();
        bin_min[i] = std::numeric_limits<float>::infinity();
    }
    if (n_bins > n_sample) n_bins = n_sample;

    const int approximate_bin_count = n_sample / n_bins;
    int current_bin_count = 1;

    // Seed bin 0 with the smallest value.
    // Hoist arr[sort_idx[0]] — used both for bin_max/bin_min init AND as the
    // initial prev_v carried across the main loop. The compiler cannot CSE
    // these indirect loads on its own because `arr` (const float *) is not
    // marked __restrict__ relative to the `int *` outputs.
    const float first_v = arr[sort_idx[0]];
    bin_max[current_bin_id] = first_v;
    bin_min[current_bin_id] = first_v;
    bin_counts[current_bin_id] += 1;

    // M4-phase-2 methodology fix: write bins[sort_idx[0]] explicitly so the
    // smallest-value position lands in bin 0. The cfuncs_fast _quantile_bins_nogil
    // does the same; the cfuncs legacy quantile_bins skipped this and relied on
    // a "stale caller-provided value" which broke parallelism + was a real bug.
    bins[sort_idx[0]] = current_bin_id;

    // Carry the previously-seen value across iterations so we drop one
    // indirect load per iter (arr[sort_idx[i-1]]). Same fp32 sequence as
    // the prior code; bit-exact.
    float prev_v = first_v;
    for (int i = 1; i < n_sample; ++i) {
        const float v = arr[sort_idx[i]];
        if (prev_v + RELATIVE_THRESHOLD < v) {
            if (current_bin_count >= approximate_bin_count) {
                current_bin_count = 0;
                current_bin_id += 1;
            }
        }
        if (current_bin_id >= n_bins) {
            current_bin_id -= 1;
        }
        bins[sort_idx[i]] = current_bin_id;
        bin_counts[current_bin_id] += 1;
        current_bin_count += 1;
        if (v > bin_max[current_bin_id]) bin_max[current_bin_id] = v;
        if (v < bin_min[current_bin_id]) bin_min[current_bin_id] = v;
        prev_v = v;
    }

    if (current_bin_id == 0) {
        is_unique = 1;
    }
    return {is_unique, current_bin_id + 1};
}

}  // namespace mttrees::cpp_kernel
