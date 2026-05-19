g TorchAO

PyTorch-native library for quantization, sparsity, and low-precision training.

## Config Classes

All configs inherit from `AOBaseConfig`. Defined in `torchao/quantization/quant_api.py`. Use `FqnToConfig` to apply different configs to different layers by module name.

## Stable vs Prototype

- **Stable** (`torchao/quantization/`, `torchao/float8/`, `torchao/sparsity/`, `torchao/optim/`): API stability guaranteed.
- **Prototype** (`torchao/prototype/`): Experimental, API may change without notice.

See [docs/source/workflows/index.md](docs/source/workflows/index.md) for the full dtype x hardware status matrix.

## Architecture and Contributing

- [Quantization Overview](docs/source/contributing/quantization_overview.rst) - full stack walkthrough, tensor subclasses, quantization flows
- [Contributor Guide](docs/source/contributing/contributor_guide.rst) - how to add tensors, kernels, configs
- [Inference Workflows](docs/source/workflows/inference.md) - which config to use for which hardware
- [PT2E Quantization](docs/source/pt2e_quantization/index.rst) - PyTorch 2 Export quantization for deployment backends (X86, XPU, ExecuTorch)

These render at https://docs.pytorch.org/ao/main/

## Deprecated APIs

Do not use or recommend these:
- `AffineQuantizedTensor` (AQT) - deleted
- `autoquant()` - deleted
- Layout registration system (`PlainLayout`, `Float8Layout`, `TensorCoreTiledLayout`, etc.) - deleted
- `TorchAODType` - deprecated
- `change_linear_weights_to_int4_woqtensors` - deleted, use `quantize_(model, Int4WeightOnlyConfig())`

New tensor types should inherit from `TorchAOBaseTensor` in `torchao/utils.py`.

## Development

```bash
# Setup
pixi run build  # 3.14 interpreter (with-GIL)
pixi run build  # 3.14t interpreter (free-threaded)

# Test (mirrors source structure, use `pixi run -e nogil pytest` for free-threaded interpreter)
pixi run pytest test/quantization/test_quant_api.py
pixi run pytest test/float8/
pixi run pytest test/prototype/mx_formats/
```


## Auditing for free-threading issues and thread-safety

With the GIL removed, Python code, pybind11 bindings, and native helper code
that previously relied on implicit serialization can now have real data races.
It is known that Triton itself still has issues, please ignore that.

What to look for

Focus on **shared mutable state and data flow**, not surface-level pattern
matching.

The key question is not "does this file mention threads?" The key question is:

- what state is shared,
- who writes it,
- who reads it,
- and can those accesses happen concurrently under free-threading?

### High-value patterns (these cause real bugs)

1. **Lazy native initialization of shared C++ state.**
   `python/src/specialize.cc` is the highest-priority example. It has lazy
   process-global initialization (`init_globals()`) guarded only by a plain
   boolean (`init_called`), plus static native caches such as
   `dtype_ptr2str`, `dtype2str`, and `type_handler_cache`. Two threads reaching
   specialization concurrently can race on initialization and on later cache
   mutation.

2. **Shared mutable caches attached to reusable Python wrapper objects.**
   `JITFunction` and `Autotuner` objects are intended to be reused across many
   calls, including calls from different threads. Mutable state hanging off
   these objects is therefore shared state. High-priority examples include:
   - `JITFunction.device_caches`
   - key/result caches used during specialization and compilation
   - `Autotuner.cache`
   - async compile bookkeeping that memoizes or finalizes work for shared keys

3. **Global registries and lazy singletons on the hot path.**
   Module-level registries and singleton objects are often safe under the GIL
   by accident and unsafe under free-threading. Important Triton examples
   include:
   - `_triton_jit_function_registry`
   - `runtime.driver.driver`
   - `backends.backends`
   - process-global `knobs.*` objects and their hook/callback state

4. **Lazy first-use state on shared Python objects.**
   `@cached_property`, ad hoc memoization, and first-use writes into object
   `__dict__` are worth attention when the object instance is shared. A good
   example is `KernelParam` in `runtime/jit.py`.

5. **Shared caches whose exact runtime guarantees are unclear.**
   Triton uses `@lru_cache` and other memoization helpers in a few places.
   Treat them as shared mutable state worth reviewing. If the exact
   free-threading behavior of a helper is unclear, leave a note for follow-up
   rather than guessing.

6. **Code running with detached thread state (GIL-released native regions).**
   `python/src/llvm.cc` and `python/src/ir.cc` contain explicit thread-state
   detach points (`py::gil_scoped_release` / `Py_BEGIN_ALLOW_THREADS`). Any
   Python C API use inside those regions is unsafe. Any native shared mutable
   state touched there is concurrently accessible.

7. **Compiled-kernel lazy finalization / lazy runtime handle init.**
   `CompiledKernel` is created by the compiler path but lazily initializes some
   runtime-facing handles on first real use. Two threads reaching that path on
   the same object are a classic free-threading audit target.

8. **Hooks/callback lists that can be mutated while the runtime is using them.**
   Triton exposes hook chains and callback slots through `knobs.py`. If one
   thread mutates these while another thread compiles, loads, or launches a
   kernel, that is shared mutable state on an active execution path.

### Pybind11 / Python-C-API specific patterns

9. **`py::call_guard<py::gil_scoped_release>()`.**
   When a binding is declared with `py::call_guard<py::gil_scoped_release>()`,
   the bound function runs with the thread state detached for that call.
   Audit the whole function body with that in mind: Python C API calls are
   forbidden, and any shared native state is concurrently accessible.

10. **Stack-based `py::gil_scoped_release allow_threads`.**
    Once constructed, the thread state is detached until the scope ends.
    Verify that no Python C API calls or unsafe shared-state access occur
    inside that region.

11. **Native containers storing `PyObject*` or Python-derived state.**
    Containers like `std::unordered_map<..., PyObject*>` are a double hazard:
    the container itself is not thread-safe, and the Python object lifetime /
    ownership assumptions may also be wrong. Verify both the container race and
    the refcount/lifetime story.

12. **Unsafe Python C APIs in thread-state-detached regions.**
    In detailed C++ passes, watch for APIs such as:
    - `PyObject_GetAttr`
    - `PyObject_Call*`
    - `PyDict_*`, `PyList_*`
    - `Py_BuildValue`, `PyArg_Parse*`
    - `Py_INCREF`, `Py_DECREF`
    - `PyErr_SetString`, `PyErr_Occurred`
    - container mutation APIs in general

    Also watch borrowed-reference APIs like `PyDict_GetItem` / `PyList_GetItem`
    whose results can become invalid if another thread mutates the underlying
    object.

### Low-value patterns (usually not worth reporting)

- **Write-once globals set during module init.**
  State created during `PYBIND11_MODULE(...)` initialization is generally
  protected by Python's import lock. Do not report it unless the code actually
  exposes concurrent access before initialization completes, or unless the
  state is later mutated.

- **C++11 magic statics.**
  Function-local `static` initialization is thread-safe by the C++ standard.
  Only flag these when the stored value itself is unsafe shared state
  (especially Python object pointers or Python-calling initializers).

- **Already-protected state.**
  Do not re-report synchronization that already exists unless you can show it is
  insufficient. For example:
  - `JITFunction._hash_lock` protects `cache_key` computation
  - `python/src/interpreter.cc` uses `std::mutex atomic_op_guard` for its
    fallback atomic operations

- **`ContextVar`-based state.**
  `_allocation._allocator` and `_async_compile.active_mode` are intentionally
  context-local. Verify that they stay context-local, but do not treat them as
  ordinary globals by default.

- **Benign debug/print flag staleness.**
  Many `knobs.*` flags control debugging, logging, or diagnostics. A stale read
  is usually not worth reporting unless it affects compilation correctness,
  cache behavior, backend/driver selection, or launch behavior.

- **Filesystem atomic replace by itself.**
  Triton's cache/build paths use filesystem operations such as atomic replace.
  That alone is not a bug. Focus on whether the surrounding shared Python/C++
  state coordinating the cache is raced.

- **The mere presence of `PyGILState_Ensure` / `PyGILState_Release`.**
  These attach/detach the calling thread's Python thread state and are required
  before calling Python C APIs, but under free-threading they do **not** provide
  mutual exclusion. The issue is when code relies on them as an implicit lock
  protecting native shared state. See GLOSSARY.md "Thread state attach / detach".

## Python built-in type thread safety

Free-threaded CPython protects most built-in container operations with
per-object critical sections, but the rules are subtle and not widely
known. **Before reporting a race on a `list`, `dict`, `set`, `bytearray`,
`memoryview`, `collections.*`, or `queue.*` object, read
[PYTHON_THREADSAFETY.md](PYTHON_THREADSAFETY.md)** — it summarizes what
is and isn't safe on free-threaded CPython, with pointers to
[https://docs.python.org/dev/library/threadsafety.html](https://docs.python.org/dev/library/threadsafety.html).

Quick rules of thumb:
- Single ops (`lst.append`, `d[k] = v`, `s.add`) on shared containers are
  safe — don't report them as corruption.
- Iteration under concurrent mutation is **not** safe, even for lists and
  dicts. The interpreter doesn't crash but the observed element sequence
  is undefined. This is the most commonly missed bug.
- Check-then-act (`if k in d: del d[k]`) and read-modify-write
  (`d[k] = d[k] + 1`) are never safe.
- `collections.defaultdict` auto-vivification (`d[missing]`) is a
  compound op and is a real race on shared defaultdicts.

