# torchao free-threading / thread-safety audit

Audit of `torchao` for races introduced by free-threaded CPython
(3.13t / 3.14t). Identify-only — no fixes proposed here. See
[CLAUDE.md](CLAUDE.md) for the audit criteria and
[PYTHON_THREADSAFETY.md](PYTHON_THREADSAFETY.md) for built-in container
rules.

## Methodology

Five passes were run as independent subagents, each scoped to one
class of pattern from CLAUDE.md:

1. **Native code** (`torchao/csrc/cpu/`, `cuda/`, `rocm/`)
2. **Module-level registries** mutated by decorators
3. **`TorchAOBaseTensor` dispatch and subclass state**
4. **Caches and lazy state** (`@lru_cache`, `@cached_property`, `if x is None:` patterns)
5. **Hooks and torch globals** (`@torch._dynamo.allow_in_graph`, `@torch.library.register_*`, `register_*hook`)

Scope follows the maintainer's choice (stable code only):
**in** — `torchao/csrc/`, `quantization/`, `float8/`, `sparsity/`, `optim/`,
`utils.py`, `core/`, `ops.py`, `swizzle/`, `dtypes/`, `kernel/`;
**out** — `prototype/`, `_models/`, `experimental/`, `testing/`, `test/`.

Severity follows CLAUDE.md's calibration:
- **HIGH** — real-world race on a hot path or correctness-affecting cache mutation.
- **MED** — race plausible but needs unusual concurrency, or hot-path read of mutation that's contractually import-only but exposed via public decorator.
- **LOW** — bookkeeping / diagnostic; race only under exotic conditions.

Patterns that CLAUDE.md classifies as low-value (write-once globals,
safe magic statics, state already protected by `std::call_once`,
import-lock-protected decorator registrations) are listed under "Items
explicitly NOT reported".

## Summary

| Section                                  | HIGH | MED | LOW |
|------------------------------------------|------|-----|-----|
| Native (`torchao/csrc/`)                 | 2    | 0   | 1   |
| Module-level registries                  | 1    | 2   | 0   |
| Tensor subclasses & dispatch             | 1    | 1   | 1   |
| Caches & lazy state                      | 4    | 3   | 1   |
| Hooks & torch globals                    | 0    | 0   | 0   |
| **Total (17 findings)**                  | **8**| **6**| **3**|

The single biggest concentration of HIGH findings is in `torchao/kernel/`
(autotuner cache + Triton lazy-init), with additional HIGH items in two
CPU ukernel selector tables (explicitly annotated "Not thread safe" in
source) and the optim 4/8-bit quantization map caches. The Python
dispatch infrastructure (`TorchAOBaseTensor`, the per-tensor-class op
tables, the `quantize_` handler registry) is mutated only at import
time in practice — flagged MED because the registration decorators are
publicly exported and a downstream caller could trigger runtime
mutation.

## Findings

### Native code (`torchao/csrc/`)

#### F-01 Unsynchronized ukernel registration table on hot path (linear_8bit_act_xbit_weight)

- **File:** `torchao/csrc/cpu/shared_kernels/linear_8bit_act_xbit_weight/kernel_selector.h:414`
- **Category:** native-cache
- **Severity:** HIGH
- **Status:** fixed — added `std::mutex` to `UKernelConfigRegistrationTable`; `register_ukernel_config` and `get_ukernel_config` now take a `std::lock_guard`
- **What:** `select_ukernel_config()` owns a function-local `static UKernelConfigRegistrationTable table` whose backing `std::unordered_map registration_table_` (line ~39) is mutated on cache miss via `register_ukernel_config(...)` (line ~429), with concurrent readers via `table.get_ukernel_config(...)` (line ~423). No mutex / atomics. Called per linear op invocation from `op_linear_8bit_act_xbit_weight-impl.h` lines 76/130/193/293/345.
- **Why it's not safe:** CLAUDE.md pattern #2/#3 ("shared mutable caches", "global registries on the hot path"). C++11 magic-static covers the *table* construction, but the stored `unordered_map` is mutated post-construction without synchronization.
- **Repro hypothesis:** Two free-threaded Python threads run the same quantized linear with a not-yet-registered `PackedWeightsHeader`. Both fail the `has_value()` check at line ~423, both enter `register_ukernel_config`, both call `registration_table_[key] = config` concurrently. Rehash/insert race → torn map, possible duplicate-key throw, or use-after-rehash crash on the reader.
- **Notes:** Source explicitly comments "Not thread safe" at line ~411. Fix surface: a `std::mutex` around `registration_table_` (or `std::call_once`-per-key).

#### F-02 Unsynchronized ukernel registration table on hot path (groupwise_lowbit_weight_lut)

- **File:** `torchao/csrc/cpu/shared_kernels/groupwise_lowbit_weight_lut/kernel_selector.h:194`
- **Category:** native-cache
- **Severity:** HIGH
- **Status:** fixed — same `std::mutex` treatment as F-01; doc comment updated from "thread-unsafe" to "thread-safe"
- **What:** Same pattern as F-01: `select_ukernel_config<weight_nbit>()` owns a function-local `static UKernelConfigRegistrationTable table` whose internal `unordered_map` is mutated by `register_ukernel_config` (line ~210) and read at lines ~202/~212 with no synchronization. Called per op from `op_groupwise_lowbit_weight_lut-impl.h` lines 61/165/219.
- **Why it's not safe:** Same as F-01 (CLAUDE.md pattern #2/#3).
- **Repro hypothesis:** Two free-threaded Python threads execute the groupwise LUT linear op concurrently before any kernel has been registered for the active CPU; both reach `register_ukernel_config` and race the map insertion.
- **Notes:** Same "Not thread safe" annotation in source.

#### F-03 Multi-instance `static Threadpool` from header inclusion

- **File:** `torchao/csrc/cpu/shared_kernels/internal/parallel-pthreadpool-impl.h:51`
- **Category:** native-static
- **Severity:** LOW
- **What:** Header defines `static Threadpool threadpool;` at file scope, so each TU including this header gets its own threadpool — but each instance is shared by all callers of `parallel_1d` within that TU. Concurrent `parallel_1d` calls on the same threadpool rely on pthreadpool's re-entrancy guarantees, which historically are *not* safe for concurrent submission from multiple submitter threads.
- **Why it's not safe:** Under the GIL only one Python thread could be in this region; free-threading removes that implicit serialization. Strictly speaking this is the wrapped library's invariant, not torchao's, but the pattern is exposed only by free-threading.
- **Repro hypothesis:** (static analysis only) Two Python threads each enter a linear op that calls `parallel_1d`; both submit work to the same `pthreadpool_t` concurrently.
- **Notes:** Verify pthreadpool's concurrent-submission semantics upstream before deciding whether torchao needs a wrapper-level lock.

### Module-level registries

#### F-04 `BEST_CONFIGS` autotuner cache — lazy singleton + RMW + pickle-while-mutate

- **File:** `torchao/kernel/autotuner.py:103, 196–242`
- **Category:** registry / cache
- **Severity:** HIGH
- **Status:** fixed — lazy init via `@functools.cache`, miss-path write + snapshot taken under `threading.Lock`, `_save_best_configs` pickles the snapshot
- **What:** Module-level `BEST_CONFIGS = None`. `get_best_config_fn` does (a) lazy init via `if BEST_CONFIGS is None: BEST_CONFIGS = _load_best_configs(); if BEST_CONFIGS is None: BEST_CONFIGS = {}` (lines ~203–208), (b) miss-path `BEST_CONFIGS[key] = (best_config, best_time)` and `_save_best_configs(BEST_CONFIGS)` (lines ~238, ~241), with reader `get_best_config_by_key` (lines ~196–198). Called per int-mm shape from `torchao/kernel/intmm_triton.py:340, 367`.
- **Why it's not safe:** Three independent unsafe patterns stacked: CLAUDE.md pattern #3 (lazy singleton), PYTHON_THREADSAFETY.md "check-then-act" (the miss-then-write), and "iteration under concurrent mutation" (`pickle.dump` walks the dict while writers may still be inserting).
- **Repro hypothesis:** Threads A and B both call `int_matmul(...)` with new but identical shapes. Both observe `key not in BEST_CONFIGS`, both autotune (wasted work), both write `BEST_CONFIGS[key] = ...`, both invoke `_save_best_configs(BEST_CONFIGS)`. The second `pickle.dump` may iterate while the other thread mutates → undefined pickled output (no crash, but the saved file content is undefined).
- **Notes:** Same shape as Triton's `Autotuner.cache` audit target in CLAUDE.md. The autotuner is only used when the Triton kernel path is selected.

#### F-05 `_QUANTIZE_CONFIG_HANDLER` public decorator allows post-import registration

- **File:** `torchao/quantization/transform_module.py:13`
- **Category:** registry
- **Severity:** MED
- **What:** Module-level `dict[type[AOBaseConfig], Callable]` populated by the public `register_quantize_module_handler` decorator (re-exported from `torchao.quantization`). Read on the `quantize_` hot path at `torchao/quantization/quant_api.py:335` and several `qat`/`sparsity` callsites.
- **Why it's not safe:** All ~40 in-tree registrations are import-time, but the decorator is publicly exported. Reads use single-key dict lookup (`_QUANTIZE_CONFIG_HANDLER[type(config)]`), which is FT-safe per PYTHON_THREADSAFETY.md, so currently the only race surface is "user plugin imports a new handler at runtime while a thread is calling `quantize_`" — and that race resolves to safe single-key dict semantics. MED because no iteration is currently performed on the read path; if iteration is introduced, this becomes HIGH immediately.
- **Repro hypothesis:** Thread A: `torchao.quantize_(model, MyCustomConfig())`. Thread B: imports a plugin whose top-level code does `@register_quantize_module_handler(SomeOtherConfig) def ...`. Thread A's lookup races thread B's `d[k]=v`; today both ops are atomic.
- **Notes:** No iteration found anywhere. The bigger question (see Open questions) is whether the API contract permits late registration.

#### F-06 `TorchAOBaseTensor._ATEN_OP_TABLE` / `_TORCH_FN_TABLE` class-level dispatch dicts

- **File:** `torchao/utils.py:425–477, 660–697, 884–911`
- **Category:** registry / dispatch
- **Severity:** MED
- **What:** Class-level `dict[type, dict[op, handler]]` on `TorchAOBaseTensor`. `__init_subclass__` (lines ~885–911) and the `_implements` / `_implements_torch_function` decorators (lines ~439, ~474) mutate `cls._ATEN_OP_TABLE[cls][op]`. Read on every `__torch_dispatch__` / `__torch_function__` call (lines ~668–691) — the per-op hot path. `__init_subclass__` contains a check-then-act (`if cls not in cls._ATEN_OP_TABLE: cls._ATEN_OP_TABLE[cls] = {}`) and an iteration via `update(parent_aten_table[parent])`.
- **Why it's not safe:** CLAUDE.md pattern #2/#3. In practice, both decorator and `__init_subclass__` mutations happen during module import or class creation — serialized by Python's per-module import lock (writes are write-once-at-import). The read path is single-key `cls._ATEN_OP_TABLE[cls][func]`, which is FT-safe (PYTHON_THREADSAFETY.md). MED because dynamic subclass creation at runtime (e.g., from a factory in user code) would expose the `update(parent_aten_table[parent])` iteration to concurrent writers.
- **Repro hypothesis:** (static analysis only) Two threads concurrently define new `TorchAOBaseTensor` subclasses. `__init_subclass__` on subclass A iterates `parent_aten_table[parent]` while subclass B's body is calling `Parent.implements(some_op)` that writes into the same dict. Iteration during concurrent insertion → undefined element sequence per PYTHON_THREADSAFETY.md.
- **Notes:** Consolidated from registry and tensor-subclass passes. No in-tree subclass factories; all 43 subclasses created at import.

### Tensor subclasses & dispatch

#### F-07 `_precomputed_scale` written on FSDP-shared Float8 weight tensor

- **File:** `torchao/float8/fsdp_utils.py:85, 174, 229–246`
- **Category:** subclass-state
- **Severity:** HIGH
- **Status:** fixed — `fsdp_pre_all_gather` now snapshots `self._precomputed_scale` into a local before the `is not None` check, eliminating the double-read TOCTOU
- **What:** `precompute_float8_dynamic_scale_for_fsdp` writes `float8_linear.weight._local_tensor._precomputed_scale = local_scale_tensor[i]` on the FSDP local tensor that is the same `WeightWithDynamicFloat8CastTensor` instance subsequently used by forward in `fsdp_pre_all_gather` (reads `self._precomputed_scale`) and `__tensor_flatten__` (line ~212).
- **Why it's not safe:** CLAUDE.md pattern #4 ("Lazy first-use state on shared Python objects"). Setting a Python instance attribute (`Optional[Tensor]`) while another thread reads it is a torn-attribute scenario under free-threading. The same tensor is also flattened/unflattened for state-dict ops; concurrent calls would interleave.
- **Repro hypothesis:** Thread A enters `fsdp_pre_all_gather`, reads `self._precomputed_scale is not None` (truthy), thread B's `precompute_float8_dynamic_scale_for_fsdp` re-writes `self._precomputed_scale = local_scale_tensor[i]`, thread A then reads `self._precomputed_scale` and uses a different tensor than the one it checked.
- **Notes:** Severity HIGH is contingent on FSDP2 producing concurrent Python execution under free-threading (see Open questions). Under the GIL the GIL was the implicit barrier; if FSDP2's overlap is purely CUDA-stream-based, the practical exposure is lower.

#### F-08 `fsdp_post_all_gather` mutates `Float8TrainingTensor._scale` slot

- **File:** `torchao/float8/fsdp_utils.py:261–266`
- **Category:** subclass-state
- **Severity:** MED
- **What:** `fsdp_post_all_gather` does `out._scale = scale` (or `out._local_tensor._scale = scale`) on the long-lived FSDP all-gather output tensor reused across iterations.
- **Why it's not safe:** `_scale` is in `Float8TrainingTensor.__slots__` (line ~281). Slot writes don't have the dict-level guarantees in PYTHON_THREADSAFETY.md, and `__setattr__` on a slot is a single store — but any other thread reading `self._scale` during matmul (e.g., backward of step *N* while post-all-gather of step *N+1* is running) sees a value that's mid-transition.
- **Repro hypothesis:** (static analysis only) FSDP overlap: thread A inside backward reads `weight._scale` for unscale, thread B's `fsdp_post_all_gather` for the next step writes the same `_scale` slot.
- **Notes:** Same FSDP-threading caveat as F-07.

#### F-09 Dead `_implements` lazy `hasattr`-check-then-create path

- **File:** `torchao/utils.py:425–428, 460–463`
- **Category:** lazy-init
- **Severity:** LOW
- **What:** Both `_implements` and `_implements_torch_function` start with `if not hasattr(cls, "_ATEN_OP_TABLE"): cls._ATEN_OP_TABLE = {}` and then `if cls not in cls._ATEN_OP_TABLE: cls._ATEN_OP_TABLE[cls] = {}`. Classic check-then-act.
- **Why it's not safe:** PYTHON_THREADSAFETY.md "check-then-act is never safe". Two threads racing to be the first decorator on a brand-new class could each see `not hasattr`, both write `cls._ATEN_OP_TABLE = {}`, and the second clobbers the first.
- **Repro hypothesis:** (static analysis only) `__init_subclass__` (line ~886) eagerly creates these dicts before any decorator can run, so for `TorchAOBaseTensor` subclasses the lazy path is unreachable. The hazard is for any consumer that calls `cls.implements(...)` on a class that *did not* go through `__init_subclass__`.
- **Notes:** Effectively dead code; report LOW for future-proofing. Consider deleting the `hasattr` branches.

### Caches & lazy state

#### F-10 `get_qmap_signed` / `get_qmap_unsigned` cache mutable list (4-bit)

- **File:** `torchao/optim/subclass_4bit.py:34–41`
- **Category:** cache
- **Severity:** HIGH
- **Status:** fixed — cached values now wrapped in `tuple(...)`
- **What:** `@lru_cache(maxsize=1)` over functions returning a Python `list[float]` from `create_dynamic_map(...)` / `torch.linspace(...).tolist()`. The same list object is returned by reference to every caller. Used at `subclass_4bit.py:128` to build `torch.tensor(qmap_list, ...)`.
- **Why it's not safe:** CLAUDE.md pattern #5 ("Shared caches whose exact runtime guarantees are unclear") + PYTHON_THREADSAFETY.md `list` rules. The `lru_cache` decorator is FT-safe in 3.13+, but the cached *value* is a mutable container shared by reference. `torch.tensor(qmap_list)` iterates the list; any concurrent caller that mutates the list (or any future code that does) → undefined iteration order.
- **Repro hypothesis:** (static analysis only) Today's callers only read the list. The race surface exists for any future code path that does `qmap_list.append(...)` or returns it to user code that mutates.
- **Notes:** Trivial fix: return a `tuple` instead of a `list`, or `return tuple(create_dynamic_map(...))`.

#### F-11 `get_qmap_signed` / `get_qmap_unsigned` cache mutable list (8-bit)

- **File:** `torchao/optim/subclass_8bit.py:30–37`
- **Category:** cache
- **Severity:** HIGH
- **Status:** fixed — cached values now wrapped in `tuple(...)`
- **What:** Identical pattern to F-10; cached `list[float]` is shared across all `OptimState8bit.zeros()` calls.
- **Why it's not safe:** Same as F-10.
- **Repro hypothesis:** (static analysis only)
- **Notes:** Fix is the same as F-10.

#### F-12 `_lazy_init_triton` unguarded boolean-flag init (blockwise_quantization)

- **File:** `torchao/kernel/blockwise_quantization.py:13–31`
- **Category:** lazy-init
- **Severity:** HIGH
- **Status:** fixed — replaced bool-flag pattern with `@functools.cache`-wrapped builder returning an immutable `_BlockwiseFp8Impls` dataclass
- **What:** Module-level `_triton_initialized = False`, `_blockwise_fp8_gemm_impl = None`, etc. `_lazy_init_triton()` sets `_triton_initialized = True` *before* assigning the kernel handles. Funneled through by every `fp8_blockwise_*` entry point (lines ~238, ~261, ~298, ~341).
- **Why it's not safe:** CLAUDE.md pattern #4. Thread A enters `_lazy_init_triton`, sets the flag, then yields during `import triton` (a slow operation). Thread B sees flag=True, returns from `_lazy_init_triton` immediately, and calls `_blockwise_fp8_gemm_impl(...)` → `_blockwise_fp8_gemm_impl is None` → `TypeError: 'NoneType' object is not callable`.
- **Repro hypothesis:** Two threads doing FP8-blockwise inference in parallel on a freshly-imported module. Thread A enters `_lazy_init_triton`, sets `_triton_initialized=True`, yields during the import. Thread B's call to `blockwise_fp8_gemm` reaches the dispatch line with `_blockwise_fp8_gemm_impl is None`.
- **Notes:** Worse than the classic double-init race because the flag is set *first*. Fix: set the flag last, or use a `threading.Lock` (or just `functools.cache` on the init function — its FT-safe locking covers exactly this).

#### F-13 `_lazy_init_triton` unguarded boolean-flag init (bsr_triton_ops)

- **File:** `torchao/kernel/bsr_triton_ops.py:449–461`
- **Category:** lazy-init
- **Severity:** HIGH
- **Status:** fixed — replaced bool-flag pattern with `@functools.cache`-wrapped builder returning an immutable `_BsrTritonImpls` dataclass
- **What:** Same shape as F-12: `_triton_initialized = False`, `_bsr_strided_addmm_kernel = None`, set flag before kernel build. Concurrent caller at line ~275 (`bsr_dense_addmm`) can see `_triton_initialized=True` but `_bsr_strided_addmm_kernel is None`.
- **Why it's not safe:** Same as F-12.
- **Repro hypothesis:** Same as F-12.
- **Notes:** Same fix.

#### F-14 `_assert_and_get_unique_device` iterates module parameters during cache miss

- **File:** `torchao/utils.py:56–71`
- **Category:** cache
- **Severity:** MED
- **What:** `@functools.cache` keyed on the module instance. The cached return value (`torch.device | None`) is immutable, but the cache-miss computation iterates `module.parameters()` and `module.buffers()` to build a `set`. If another thread is concurrently mutating the module (adding parameters/buffers, `register_buffer`, `add_module`), the iteration races.
- **Why it's not safe:** PYTHON_THREADSAFETY.md "iteration under concurrent mutation". The race window is the first call only; after the cache fills, the cached value is safe.
- **Repro hypothesis:** Thread A calls `benchmark_model(model)` triggering first-fill iteration of `model.parameters()`. Thread B concurrently calls `model.register_buffer(...)`.
- **Notes:** Modules are typically constructed before any inference thread starts, so this is mostly theoretical — but the function is also called from inside hot paths (e.g., `quantize_`).

#### F-15 `_register_quantization_weight_pack_pass` side-effect-only `@lru_cache(None)`

- **File:** `torchao/quantization/pt2e/inductor_passes/x86.py:3457–3482`
- **Category:** cache
- **Severity:** MED
- **What:** `@functools.lru_cache(None)` over a function whose body performs `_register_*` side effects (registering inductor passes globally). Return value is `None`; the cache is used as a "run-once" idempotency guard.
- **Why it's not safe:** CLAUDE.md pattern #5 ("shared caches whose exact runtime guarantees are unclear"). Under CPython 3.13+, `lru_cache` is FT-safe enough that concurrent callers won't both execute the body — but the registered passes call into PyTorch inductor's process-global pattern-matcher registry, whose FT safety is upstream's concern.
- **Repro hypothesis:** (static analysis only)
- **Notes:** Verify that `lru_cache` provides exact single-execution semantics for side-effect-only bodies under 3.13t/3.14t.

#### F-16 PT2E observer lazy `block_size` / `original_dtype` write on shared instance

- **File:** `torchao/quantization/pt2e/observer.py:1683–1684`; writes at `torchao/quantization/pt2e/_affine_quantization.py:633–635, 724–726, 806–807`
- **Category:** lazy-init
- **Severity:** MED
- **What:** `AffineQuantizedObserverBase.__init__` sets `self.block_size = None`, `self.original_dtype = None`. `forward()` writes these (`self.original_dtype = input.dtype`, `self.block_size = get_block_size(...)`) and does `self.min_val.copy_(min_val)` (RMW). The observer is attached to a module and shared across forward calls.
- **Why it's not safe:** CLAUDE.md pattern #4. Two concurrent forwards on the same observer with different input dtypes/shapes can leave `block_size` and `original_dtype` from one thread paired with `min_val` from another.
- **Repro hypothesis:** Calibration-style: two threads call `model.forward(x)` on the same model with different `x.dtype` or `x.shape` — observer state is interleaved inconsistently.
- **Notes:** PTQ calibration is typically single-threaded by convention; the code does not enforce this.

#### F-17 `_cur_fqn` / `_fqn_to_op_to_shape_to_count` diagnostic globals

- **File:** `torchao/quantization/utils.py:71, 88–90, 98–113`
- **Category:** cache (diagnostic)
- **Severity:** LOW
- **What:** Module-level `_cur_fqn` set by forward hooks installed by `_apply_logging_hook`, read inside `LoggingTensorMode.__torch_dispatch__`. Nested `_fqn_to_op_to_shape_to_count` dict is auto-vivified via check-then-act, and counts are incremented via `d[k][j][s] += 1` (RMW).
- **Why it's not safe:** PYTHON_THREADSAFETY.md "check-then-act / read-modify-write are never safe".
- **Repro hypothesis:** (static analysis only) `_apply_logging_hook` has no in-scope callers; source already comments "not safe for any kind of multithreading" at line ~70.
- **Notes:** Diagnostic-only path; severity LOW under CLAUDE.md "Benign debug/print flag staleness". Consolidates findings from pass 4 and pass 5.

### Hooks & torch globals

No HIGH or MED findings. Every `@torch._dynamo.allow_in_graph`,
`@torch.library.register_*`, `torch.library.Library(...)`, and
`register_buffer` callsite in stable code is fired at module import or
during the user's own module construction. Per CLAUDE.md's "write-once
globals" carve-out, these are not reported. The one diagnostic-path
case is F-17 above (covered under Caches).

## Items explicitly NOT reported

Per CLAUDE.md "Low-value patterns" — these were considered and
dismissed:

### Native (csrc/)

- `torchao/csrc/cuda/cutlass_extensions/common.h:27–65` — `device_flags` + `device_properties` sized once via magic-static lambda, per-element writes guarded by per-device `std::call_once`. Protection sufficient.
- `torchao/csrc/rocm/swizzle/swizzle.cpp:197` — function-local magic static for `workspace_size` (POD).
- `torchao/csrc/cpu/aten_kernels/float8_linear_krnl.cpp:22–36, 561`, `da8w4_linear_krnl.cpp:13–22, 714`, `quantized_sdpa_krnl.cpp:34–54` — CPUBLAS pack flags protected by `std::call_once` on POD bools; protection sufficient.
- `torchao/csrc/cpu/aten_kernels/dispatch.cpp:143–192` — `dispatch_str`, `dispatch_debug`, `dispatch_mode`, `dispatcher`, and the `dispatch_mode_map` are initialized within a single `std::call_once`; subsequent reads are write-once values.
- `torchao/csrc/cpu/aten_kernels/utils.h:114` — `inline int dispatch_mode = -1` written only inside the `call_once` lambda; effectively write-once.
- Zero hits in `torchao/csrc/` for `pybind11`, `Py_BEGIN_ALLOW_THREADS`, `py::gil_scoped_release`, `PyGILState_Ensure`, `PyObject`, `call_guard`. The native side has no Python C API surface.

### Registries

- `torchao/float8/float8_ops.py:21` `FLOAT8_OPS_TABLE` — `@implements` decorator file-local, all mutation at import. Read uses single-key `in`/`[]`.
- `torchao/swizzle/swizzle_ops.py:14` `SWIZZLE_OPS_TABLE` — same shape as above.
- `torchao/quantization/quantize_/workflows/nf4/nf4_tensor.py:66, 1092` `NF4_OPS_TABLE`, `NF4_TORCH_FUNCTIONS` — same shape.
- `torchao/sparsity/training/pointwise_ops.py:86` `CUTLASS_POINTWISE_OP_DISPATCH_TABLE` — module-level dict literal, never mutated.
- `torchao/core/config.py:191` `ALLOWED_AO_MODULES` — module-level `set` literal, never mutated.
- `torchao/quantization/quant_primitives.py:218` `_ONES_TABLE` — module-level literal, never mutated.
- `torchao/quantization/pt2e/*` various `_QUANT_OPS`, `_QUANTIZE_OPS`, `_DEQUANTIZE_OPS`, `_EQUIVALENT_TYPES_DICT` — built once at import, never mutated.
- `torchao/optim/subclass_{4bit,8bit,fp8}.py` `_optim_state_*_c10d_ops` — list `.append`'d once at import based on torch version, then frozen.

### Tensor subclasses

- `Int4Tensor`, `Int8Tensor`, `Float8Tensor`, `IntxOpaqueTensor`, `BlockSparseTensor` — instance attributes set only in `__init__`; dispatch handlers re-construct rather than mutate. Write-once pattern.
- `OptimState4bit`, `OptimState8bit` `__init__` — same; `qmap` is shared via the `@lru_cache` factory (see F-10/F-11), and `copy_` semantics on the optimizer-state subclasses are documented in-place writes (PyTorch optimizer-state contract, not torchao bug).
- `Float8TrainingTensor.__slots__` (`float8_training_tensor.py:281–288`) — `__slots__` use avoids `__dict__` races; the specific slot mutation in FSDP is F-08.
- `nf4_tensor.py:675–697` `NF4Tensor.from_tensor` — builds the 16-entry NF4 lookup tensor on every call (the misleading comment about "caching on class def" is *not* implemented; no race today).
- `torchao/optim/subclass_{4bit,8bit,fp8}.py`, `float8/fsdp_utils.py`, etc. — `torch.serialization.add_safe_globals([...])` at import. Write-once.

### Caches

- `torchao/ops.py:65` `cached_compute_capability` — `@lru_cache` returning `int` (immutable). FT-safe cache + immutable value.
- `torchao/ops.py:293` `_get_dtypes` — `@lru_cache` returning tuple of dtype objects (immutable).
- `torchao/quantization/pt2e/quantizer/{x86_inductor_quantizer,xpu_inductor_quantizer,arm_inductor_quantizer}.py` `get_default_*_inductor_quantization_config` — `@lru_cache` returning frozen `QuantizationConfig` dataclass.
- `torchao/optim/cpu_offload.py:50` `self.d_opt = None` — set in `__init__`, resolved before return; one-shot per-instance.
- `torchao/quantization/qat/linear.py:91, 106`, `qat/embedding.py:72` `self.activation_fake_quantizer = None` — conditional per-instance attribute, not lazy init.
- `torchao/float8/float8_tensor_parallel.py:210, 256–257` `self.linear_mm_config = None` — one-time TP setup.

### Hooks

- All `torch.library.Library(...)`, `@register_fake`, `@custom_op`, `@register_kernel`, `@torch._dynamo.allow_in_graph` in `torchao/ops.py`, `quantization/quant_primitives.py`, `quantization/pt2e/reference_representation_rewrite.py`, `quantization/quantize_/workflows/float8/...`, `float8/float8_{training_tensor,linear,scaling_utils}.py`, `sparsity/training/autograd.py`, `quantization/quantize_/workflows/nf4/nf4_tensor.py:1205` — all decorators fire at import. Import lock serializes.
- `torchao/optim/cpu_offload.py:105` `p_device.register_post_accumulate_grad_hook(backward_hook)` — registration on the user's own parameter tensors during `__init__`. Per-parameter, one-shot.
- `torchao/sparsity/training/__init__.py:13` `SparseSemiStructuredTensorCUTLASS._load_dispatch_table()` — write-once at import into an upstream-torch class.
- All `register_buffer(...)` and `register_*_hook(...)` calls on torchao module subclasses — registered during module `__init__` on per-instance state.

## Open questions

These could not be resolved by static analysis. Resolving them would
change the severity calibration on several HIGH/MED findings:

- **FSDP2 threading model under free-threading.** F-07 and F-08 are calibrated HIGH/MED assuming FSDP2 can produce concurrent Python execution against the same wrapper-tensor instance. Under the GIL the GIL was the implicit barrier; if FSDP2's overlap is purely CUDA-stream-based with single-threaded Python orchestration, the practical exposure of `_precomputed_scale` and `_scale` slot writes is much lower. Recommend reading `torch.distributed._composable.fsdp` source under 3.13t/3.14t.
- **`@register_quantize_module_handler` contract.** F-05 is MED because reads are single-key. If torchao publicly documents this decorator as a runtime extension point (plugins / quantize-time registration), it should be either (a) gated to import-time only, or (b) protected by a `threading.Lock` plus a snapshot-style read. Recommend clarifying the API contract.
- **`lru_cache` exact semantics under 3.13t/3.14t for side-effect-only bodies.** F-15 assumes CPython's per-`lru_cache` lock serializes the body. The porting guides say the cache is FT-safe; whether the inner call is *executed exactly once* across concurrent first-callers needs verification.
- **`BEST_CONFIGS` reachability in production.** F-04 (HIGH) is via the Triton autotuner path in `torchao/kernel/`. Whether this path is on a per-process hot path or only an offline/one-shot tuning path determines real-world severity.
- **`TorchAOBaseTensor` subclass factories.** F-06 is MED because no in-tree dynamic subclass creation exists. If torchao publicly documents creating subclasses at runtime (e.g., a "tensor wrapper factory"), severity rises immediately because `__init_subclass__` iterates the parent's `_ATEN_OP_TABLE[parent]` during `update(...)`.
- **`_apply_logging_hook` callers.** F-17 is LOW today because the function has no in-scope callers. If it is publicly exposed and used, the diagnostic globals become a real race.
- **pthreadpool concurrent-submission semantics.** F-03 depends on upstream pthreadpool's guarantees. Historical behavior is "not safe to submit from multiple threads on the same pool"; verify against the bundled pthreadpool version.
