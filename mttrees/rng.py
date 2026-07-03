"""Deterministic per-tree RNG contract for the fast backend.

Establishes the M0 contract: a single `base_seed` (int) generates a
deterministic per-tree seed sequence via numpy's `SeedSequence.spawn`.
Each tree's seed is propagated into:
  - Python-level bagging draw (`features.sample(random_state=seed)`)
  - Cython-level feature subsampling (which consumes the global NumPy RNG
    inside `sample_features`; we seed it with `np.random.seed(seed)` just
    before each kernel call).

This is a property of the Python wrappers, not the kernels themselves;
the legacy `.pyx` files stay byte-identical.
"""

import numpy as np


def per_tree_seeds(base_seed, n_trees):
    """Return n_trees uint32 seeds derived deterministically from base_seed.

    Two SeedSequence trees rooted at the same base_seed produce identical
    output regardless of host or numpy version, so this contract is portable.
    """
    if base_seed is None:
        base_seed = 0
    ss = np.random.SeedSequence(int(base_seed))
    children = ss.spawn(int(n_trees))
    return [int(c.generate_state(1, dtype=np.uint32)[0]) for c in children]


def seed_numpy_global(seed):
    """Seed numpy's global RNG (consumed by Cython kernels' sample_features).

    Returns the prior state so callers can restore it after the kernel call
    if they want to avoid side effects on the caller's global stream.
    """
    state = np.random.get_state()
    np.random.seed(int(seed))
    return state


def restore_numpy_global(state):
    np.random.set_state(state)
