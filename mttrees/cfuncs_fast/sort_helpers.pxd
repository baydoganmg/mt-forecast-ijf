# Inline-header sort primitives shared by the prebin and selfhash fast kernels.
#
# All routines are stable: equal keys preserve their input order. This matches
# the legacy `np.argsort(..., kind='stable')` semantics and is required for
# the tied-split structural equivalence assertion in the O7 oracle.
#
# Algorithm: bottom-up merge sort on an integer index array, keyed by either a
# 1D float view or a column of a 2D float view. Scratch buffer the same size
# as the index array is supplied by the caller (avoids any allocation in the
# hot path, and makes the routines nogil-safe).
#
# Cost: O(N log N) work, O(N) extra memory. For our typical N = n_predictor *
# n_bins (~640 at B=32, P=20), this is ~6 thousand comparisons per objective
# per split, vs the legacy np.argsort path which round-trips out to Python
# under the GIL once per call.

cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline void _stable_msort_by_col_f32(
        float[:, :] data, int N, int col,
        int* indices, int* scratch) nogil:
    """Sort indices[0:N] stably by data[indices[i], col] ascending."""
    cdef int width, i, l, m, r, a, b, k
    cdef float va, vb

    for i in range(N):
        indices[i] = i

    width = 1
    while width < N:
        i = 0
        while i < N:
            l = i
            m = i + width
            if m > N:
                m = N
            r = i + 2 * width
            if r > N:
                r = N
            a = l
            b = m
            k = l
            while a < m and b < r:
                va = data[indices[a], col]
                vb = data[indices[b], col]
                # Stable: take left when equal (vb < va is strict).
                if vb < va:
                    scratch[k] = indices[b]
                    b += 1
                else:
                    scratch[k] = indices[a]
                    a += 1
                k += 1
            while a < m:
                scratch[k] = indices[a]
                a += 1
                k += 1
            while b < r:
                scratch[k] = indices[b]
                b += 1
                k += 1
            for k in range(l, r):
                indices[k] = scratch[k]
            i += 2 * width
        width *= 2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline void _stable_msort_1d_f32(
        float[:] arr, int N,
        int* indices, int* scratch) nogil:
    """Sort indices[0:N] stably by arr[indices[i]] ascending."""
    cdef int width, i, l, m, r, a, b, k
    cdef float va, vb

    for i in range(N):
        indices[i] = i

    width = 1
    while width < N:
        i = 0
        while i < N:
            l = i
            m = i + width
            if m > N:
                m = N
            r = i + 2 * width
            if r > N:
                r = N
            a = l
            b = m
            k = l
            while a < m and b < r:
                va = arr[indices[a]]
                vb = arr[indices[b]]
                if vb < va:
                    scratch[k] = indices[b]
                    b += 1
                else:
                    scratch[k] = indices[a]
                    a += 1
                k += 1
            while a < m:
                scratch[k] = indices[a]
                a += 1
                k += 1
            while b < r:
                scratch[k] = indices[b]
                b += 1
                k += 1
            for k in range(l, r):
                indices[k] = scratch[k]
            i += 2 * width
        width *= 2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline void fill_stable_rank_2d_axis0_f32(
        float[:, :] data, int N, int M,
        int[:, :] out_rank,
        int* idx_buf, int* scratch) nogil:
    """For each column o in [0, M), write
       out_rank[i, o] = stable rank of data[i, o] among data[:, o].

    Drop-in replacement for the legacy double-argsort pattern:
        sort_idx = np.argsort(data, axis=0, kind='stable')
        rank    = np.argsort(sort_idx, axis=0, kind='stable')

    Faster because it sorts once per column (M sorts of size N) and
    writes the rank directly, instead of two sorts of size N x M.
    """
    cdef int o, k
    for o in range(M):
        _stable_msort_by_col_f32(data, N, o, idx_buf, scratch)
        for k in range(N):
            out_rank[idx_buf[k], o] = k


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline void stable_argsort_1d_f32(
        float[:] arr, int N,
        int[:] out_idx, int* scratch) nogil:
    """Drop-in replacement for `np.argsort(arr, kind='stable')` over fp32 1D."""
    cdef int width, i, l, m, r, a, b, k
    cdef float va, vb

    for i in range(N):
        out_idx[i] = i

    width = 1
    while width < N:
        i = 0
        while i < N:
            l = i
            m = i + width
            if m > N:
                m = N
            r = i + 2 * width
            if r > N:
                r = N
            a = l
            b = m
            k = l
            while a < m and b < r:
                va = arr[out_idx[a]]
                vb = arr[out_idx[b]]
                if vb < va:
                    scratch[k] = out_idx[b]
                    b += 1
                else:
                    scratch[k] = out_idx[a]
                    a += 1
                k += 1
            while a < m:
                scratch[k] = out_idx[a]
                a += 1
                k += 1
            while b < r:
                scratch[k] = out_idx[b]
                b += 1
                k += 1
            for k in range(l, r):
                out_idx[k] = scratch[k]
            i += 2 * width
        width *= 2
