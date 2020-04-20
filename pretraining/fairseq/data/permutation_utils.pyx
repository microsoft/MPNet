i# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

mport numpy as np

cimport cython
cimport numpy as np

DTYPE = np.int64
ctypedef np.int64_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray make_span_perm(np.ndarray perm, list word_idx, int n):
    cdef np.ndarray ans
    cdef int i, g, j, start, end
    ans = np.zeros(n, dtype=np.int64)

    g = 0
    for i in range(len(word_idx) - 1):
        start = word_idx[perm[i]]
        end = word_idx[perm[i] + 1]
        for j in range(start, end):
            ans[g] = j
            g = g + 1
    return ans
