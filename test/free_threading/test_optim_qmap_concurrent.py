# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""Free-threading regression tests for OptimState{4,8}bit qmap cache.

THREADING_AUDIT.md findings F-10 and F-11: the `@lru_cache` factories
`get_qmap_signed` / `get_qmap_unsigned` previously returned a `list[float]`
shared by reference across all callers. Concurrent `torch.tensor(qmap_list, ...)`
calls would iterate the same list, and any future caller that mutated it would
trigger an undefined iteration order under free-threaded CPython
(PYTHON_THREADSAFETY.md: list iteration under concurrent mutation is unsafe).

The fix returns an immutable `tuple` from the cached factories. These tests
defend that contract end-to-end.
"""

import threading

import pytest
import torch

from torchao.optim.subclass_4bit import (
    OptimState4bit,
)
from torchao.optim.subclass_4bit import (
    get_qmap_signed as get_qmap_signed_4bit,
)
from torchao.optim.subclass_4bit import (
    get_qmap_unsigned as get_qmap_unsigned_4bit,
)
from torchao.optim.subclass_8bit import (
    OptimState8bit,
)
from torchao.optim.subclass_8bit import (
    get_qmap_signed as get_qmap_signed_8bit,
)
from torchao.optim.subclass_8bit import (
    get_qmap_unsigned as get_qmap_unsigned_8bit,
)


@pytest.mark.parametrize(
    "getter",
    [
        get_qmap_signed_4bit,
        get_qmap_unsigned_4bit,
        get_qmap_signed_8bit,
        get_qmap_unsigned_8bit,
    ],
    ids=[
        "signed_4bit",
        "unsigned_4bit",
        "signed_8bit",
        "unsigned_8bit",
    ],
)
def test_qmap_getter_returns_tuple(getter):
    # Regression guard for F-10/F-11: the cached value must be immutable.
    result = getter()
    assert isinstance(result, tuple), (
        f"{getter.__name__} returned {type(result).__name__}, expected tuple. "
        "Reverting to list would re-introduce the shared-mutable-cache race."
    )


@pytest.mark.parametrize("signed", [True, False])
@pytest.mark.parametrize(
    "cls,block_size,shape",
    [
        (OptimState4bit, 128, (128,)),
        (OptimState8bit, 256, (256,)),
    ],
    ids=["4bit", "8bit"],
)
def test_concurrent_zeros_no_qmap_corruption(cls, block_size, shape, signed):
    """N threads concurrently building OptimState{4,8}bit.zeros() must not
    observe a corrupted or torn qmap tensor.

    Pre-fix (cached list): if any caller mutated the shared list, concurrent
    `torch.tensor(qmap_list)` calls would observe interleaved elements.
    Post-fix (cached tuple): the value is immutable, so iteration cannot tear.
    """
    if cls is OptimState4bit:
        ref_seq = get_qmap_signed_4bit() if signed else get_qmap_unsigned_4bit()
    else:
        ref_seq = get_qmap_signed_8bit() if signed else get_qmap_unsigned_8bit()
    reference = torch.tensor(ref_seq, dtype=torch.float32)

    n_threads = 16
    n_iters = 50
    errors: list = []
    barrier = threading.Barrier(n_threads)

    def worker():
        barrier.wait()
        try:
            for _ in range(n_iters):
                state = cls.zeros(shape, signed=signed, block_size=block_size)
                if not torch.equal(state.qmap, reference):
                    errors.append(
                        ("qmap mismatch", state.qmap.tolist(), reference.tolist())
                    )
        except Exception as e:
            errors.append(("exception", type(e).__name__, str(e)))

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
