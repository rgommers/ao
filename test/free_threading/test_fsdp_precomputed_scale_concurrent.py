# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""Free-threading regression test for `WeightWithDynamicFloat8CastTensor._precomputed_scale`.

THREADING_AUDIT.md finding F-07: `fsdp_pre_all_gather` previously read
`self._precomputed_scale` twice — once for the `is not None` check and once
for the actual use. Under free-threaded CPython,
`precompute_float8_dynamic_scale_for_fsdp` can re-assign the attribute between
the two reads (CLAUDE.md pattern #4 + PYTHON_THREADSAFETY.md "check-then-act").
The fix snapshots `self._precomputed_scale` into a local once and branches on
the local.

This test exercises the snapshot pattern: a writer thread oscillates
`_precomputed_scale` between a real tensor and `None`; reader threads call
`fsdp_pre_all_gather(mesh=None)` repeatedly. Without the fix, a reader can
pass `None` to `hp_tensor_and_scale_to_float8` and crash with a TypeError.
With the fix, every reader observes a consistent branch.

Note: under `pixi run -e nogil pytest`, triton's transitive import re-enables
the GIL. To exercise the actual free-threaded path, run with
`PYTHON_GIL=0 pixi run -e nogil pytest test/free_threading/`.
"""

import threading

import pytest
import torch

from torchao.float8.float8_training_tensor import LinearMMConfig
from torchao.float8.fsdp_utils import WeightWithDynamicFloat8CastTensor


def _make_weight():
    base = torch.randn(64, 64, dtype=torch.float32)
    return WeightWithDynamicFloat8CastTensor(
        tensor=base,
        linear_mm_config=LinearMMConfig(),
        dtype=torch.float8_e4m3fn,
    )


def test_fsdp_pre_all_gather_works_both_branches():
    # Sanity check: both branches reachable with mesh=None on CPU.
    w = _make_weight()
    assert w._precomputed_scale is None
    # Dynamic branch
    data, scale = w.fsdp_pre_all_gather(mesh=None)
    assert data[0].shape == (64, 64)
    assert scale[0].numel() == 1
    # Precomputed branch
    w._precomputed_scale = torch.tensor(1.0)
    data, scale = w.fsdp_pre_all_gather(mesh=None)
    assert data[0].shape == (64, 64)
    assert torch.equal(scale[0], torch.tensor(1.0))


@pytest.mark.parametrize("n_readers", [4, 8])
def test_concurrent_precomputed_scale_write_and_pre_all_gather(n_readers):
    """One writer oscillates `_precomputed_scale` between a tensor and None;
    multiple readers call `fsdp_pre_all_gather`. The reader must never see a
    torn state where it took the `is not None` branch but `precomputed_scale`
    is `None` (which would crash `hp_tensor_and_scale_to_float8`).
    """
    w = _make_weight()

    n_reader_iters = 1000
    stop = threading.Event()
    errors: list = []
    barrier = threading.Barrier(n_readers + 1)

    def writer():
        barrier.wait()
        i = 0
        # Oscillate quickly to maximize the chance of catching a torn read.
        while not stop.is_set():
            w._precomputed_scale = torch.tensor(1.0 + (i % 1000) * 1e-6)
            i += 1
            if i % 8 == 0:
                w._precomputed_scale = None

    def reader():
        barrier.wait()
        try:
            for _ in range(n_reader_iters):
                data, scale = w.fsdp_pre_all_gather(mesh=None)
                # If the snapshot fix is in place, scale[0] is always a real
                # tensor with one element. Under the pre-fix bug, the reader
                # could pass None to hp_tensor_and_scale_to_float8, raising
                # TypeError / AttributeError inside that call.
                if scale[0] is None or scale[0].numel() != 1:
                    errors.append(("torn scale", scale))
        except Exception as e:  # surface race-induced exceptions
            errors.append(("exception", type(e).__name__, str(e)))

    threads = [threading.Thread(target=writer)] + [
        threading.Thread(target=reader) for _ in range(n_readers)
    ]
    for t in threads:
        t.start()
    # Let readers finish their fixed iteration count, then stop the writer.
    for t in threads[1:]:
        t.join()
    stop.set()
    threads[0].join()
    assert not errors, errors
