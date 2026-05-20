# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""Free-threading regression tests for CPU shared-kernels' UKernelConfigRegistrationTable.

THREADING_AUDIT.md findings F-01 (`linear_8bit_act_xbit_weight`) and F-02
(`groupwise_lowbit_weight_lut`) flagged a function-local `static`
`UKernelConfigRegistrationTable` whose backing `std::unordered_map` was
mutated without synchronization on cache miss and read concurrently on the
hot path. The fix adds a `mutable std::mutex` and `std::lock_guard` in both
`register_ukernel_config` and `get_ukernel_config`.

Both registration tables are populated only inside
`#if defined(TORCHAO_BUILD_CPU_AARCH64)`. On x86 the op is not registered
at all (`hasattr(torch.ops.torchao, "_linear_8bit_act_4bit_weight")` is
`False`), so these tests skip cleanly on this dev box. On aarch64 they
exercise the mutex on the read path (concurrent `get_ukernel_config`).

**Limitations (worth understanding before relying on these tests):**

- The `UKernelConfigRegistrationTable` is a function-local `static`,
  populated once per process. After the first call in a pytest session,
  subsequent calls only hit `get_ukernel_config` (the read path), not
  `register_ukernel_config` (the write path). To exercise the
  insert-race window we would need to fork a fresh subprocess per
  parametrization. That is out of scope for this scaffold; the
  hot-path mutex is still exercised here and that is the dominant
  surface in practice.
- These tests are **scaffolded but unverified on aarch64 by the original
  author**. The signatures used match the schemas declared in
  `op_linear_8bit_act_xbit_weight_aten.cpp` and
  `op_groupwise_lowbit_weight_lut_aten.cpp`; if a kernel rejects the
  shapes here, fix the inputs and update the comment when running on
  aarch64.

Run under `pixi run -e nogil pytest test/free_threading/` on aarch64 with
`PYTHON_GIL=0` to actually exercise the free-threaded code path.
"""

import platform
import threading

import pytest
import torch


_IS_AARCH64 = platform.machine() in ("aarch64", "arm64")


def _have_linear_8bit_op() -> bool:
    return hasattr(torch.ops.torchao, "_linear_8bit_act_4bit_weight")


def _have_groupwise_lut_op() -> bool:
    return hasattr(torch.ops.torchao, "_linear_groupwise_4bit_weight_with_lut")


@pytest.mark.skipif(
    not _IS_AARCH64,
    reason="UKernelConfigRegistrationTable for linear_8bit_act_xbit_weight is "
    "only populated under TORCHAO_BUILD_CPU_AARCH64",
)
@pytest.mark.skipif(
    not _have_linear_8bit_op(),
    reason="torchao::_linear_8bit_act_4bit_weight not registered in this build",
)
@pytest.mark.parametrize("n_threads", [4, 8])
def test_linear_8bit_act_4bit_weight_concurrent(n_threads):
    """F-01: hammer `_pack_8bit_act_4bit_weight` and `_linear_8bit_act_4bit_weight`
    from N threads to exercise concurrent reads against the
    UKernelConfigRegistrationTable.

    Schema (from `op_linear_8bit_act_xbit_weight_aten.cpp`):
        _pack_8bit_act_4bit_weight(
            Tensor weight_qvals, Tensor weight_scales, Tensor? weight_zeros,
            int group_size, Tensor? bias, str? target) -> Tensor
        _linear_8bit_act_4bit_weight(
            Tensor activations, Tensor packed_weights,
            int group_size, int n, int k) -> Tensor
    """
    n_out, k = 8, 64
    group_size = 32
    m = 4

    weight_qvals = torch.randint(-8, 7, (n_out, k), dtype=torch.int8)
    weight_scales = (
        torch.randn(n_out * (k // group_size), dtype=torch.float32).abs() + 0.01
    )

    # Pack once to get a canonical packed-weight blob; subsequent threads
    # only call the read-path `_linear_*` op against this packed buffer.
    packed = torch.ops.torchao._pack_8bit_act_4bit_weight(
        weight_qvals, weight_scales, None, group_size, None, None
    )

    activations = torch.randn(m, k, dtype=torch.float32)
    reference = torch.ops.torchao._linear_8bit_act_4bit_weight(
        activations, packed, group_size, n_out, k
    )

    errors: list = []
    results: list = [None] * n_threads
    barrier = threading.Barrier(n_threads)

    def worker(idx: int):
        barrier.wait()
        try:
            for _ in range(50):
                out = torch.ops.torchao._linear_8bit_act_4bit_weight(
                    activations, packed, group_size, n_out, k
                )
                if not torch.allclose(out, reference):
                    errors.append(("mismatch", idx))
                    return
            results[idx] = out
        except Exception as e:  # race-induced crash inside select_ukernel_config
            errors.append(("exception", idx, type(e).__name__, str(e)))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors


@pytest.mark.skipif(
    not _IS_AARCH64,
    reason="UKernelConfigRegistrationTable for groupwise_lowbit_weight_lut is "
    "only populated under TORCHAO_BUILD_CPU_AARCH64",
)
@pytest.mark.skipif(
    not _have_groupwise_lut_op(),
    reason="torchao::_linear_groupwise_4bit_weight_with_lut not registered in this build",
)
@pytest.mark.skip(
    reason="Scaffold only: input shapes for _pack_groupwise_4bit_weight_with_lut "
    "(weight_qval_idxs, luts) need to be verified against a known-good "
    "GroupwiseLutWeightConfig pipeline on aarch64. The mutex pattern in F-02 is "
    "identical to F-01; this placeholder documents the test the next reader "
    "should fill in."
)
@pytest.mark.parametrize("n_threads", [4, 8])
def test_groupwise_lowbit_weight_lut_concurrent(n_threads):
    """F-02: structurally identical to F-01 but exercises a different
    `UKernelConfigRegistrationTable` instance (in the
    `torchao::ops::groupwise_lowbit_weight_lut` namespace).

    Schema (from `op_groupwise_lowbit_weight_lut_aten.cpp`):
        _pack_groupwise_4bit_weight_with_lut(
            Tensor weight_qval_idxs, Tensor luts,
            int scale_group_size, int lut_group_size,
            Tensor? weight_scales, Tensor? bias, str? target) -> Tensor
        _linear_groupwise_4bit_weight_with_lut(
            Tensor activations, Tensor packed_weights,
            int scale_group_size, int lut_group_size, int n, int k) -> Tensor

    A working set of inputs can be derived by tracing a single forward
    pass of `quantize_(model, GroupwiseLutWeightConfig(...))` on aarch64
    (see `test/prototype/test_groupwise_lowbit_weight_lut_quantizer.py`).
    Once those are in hand, drop the `@pytest.mark.skip` decorator and
    fill in `weight_qval_idxs` / `luts` / `weight_scales` accordingly.
    """
    # Skeleton (intentionally not executed; @pytest.mark.skip above).
    raise NotImplementedError("placeholder — see docstring")
