# Python built-in type thread safety (free-threaded CPython)

Reference notes for auditing Triton under Python 3.13t / 3.14t and later.
The authoritative source is
[https://docs.python.org/dev/library/threadsafety.html](https://docs.python.org/dev/library/threadsafety.html).
Consult it whenever you are unsure — this file is a summary, not a spec.

Further reading for porting existing code and extensions to free-threading:

- [Porting Python code to free-threading](https://py-free-threading.github.io/porting/)
  — practical guide for pure-Python porting: common race patterns,
  synchronization primitives, lazy initialization, and `@functools.lru_cache`.
- [Porting C extensions to free-threading](https://py-free-threading.github.io/porting-extensions/)
  — pybind11 / C API guidance: critical sections, `Py_MOD_GIL_NOT_USED`,
  thread-unsafe C APIs, and data race debugging with TSan.

Free-threaded CPython protects most built-in container operations with
per-object critical sections. This is poorly known because free-threading
is new, so auditors — including LLMs — often get these rules wrong in
both directions.

**The critical rule:** single operations are atomic or safe; **compound
operations, check-then-act patterns, and iteration under concurrent
mutation are not**. When in doubt, iteration is almost never safe against
a concurrent writer.

## `list`

Safe on shared lists:
- `lst[i]` and `lst[i] = x` (single element read/write)
- `lst.append(x)`, `lst.pop()` (without index), `lst.clear()`
- `lst.copy()`, `lst1 + lst2`
- `len(lst)`

Lock-free, may observe concurrent modifications (results are well-defined
element-by-element but the overall answer may reflect a mid-mutation state):
- `item in lst`, `lst.index(item)`, `lst.count(item)`

**Not thread-safe** — do not assume these are safe and do flag them
when you see them on shared lists:
- `for item in lst: ...` with any concurrent writer (including `append`).
  The *interpreter does not crash*, but the observed element sequence is
  undefined — existing elements may be skipped or revisited. Fix: snapshot
  under a lock, iterate the snapshot outside the lock.
- `lst[i] = lst[i] + 1` (read-modify-write).
- `if lst: item = lst.pop()` (check-then-act).
- `lst.insert(idx, x)` and `lst.pop(idx)` for non-end indices: shift
  multiple elements and can let lock-free readers observe intermediate
  states.
- `lst.remove(x)` — `__eq__` can execute arbitrary Python code, so the
  list can be concurrently modified during the search.

## `dict`

Safe on shared dicts (single key/value operations):
- `d[key]`, `d.get(key, default)`, `key in d`, `len(d)`
- `d[key] = value`, `del d[key]`, `d.pop(key)`, `d.popitem()`, `d.setdefault(key, v)`
- `d.copy()`, `d | other`, `d.update(other)` when `other` is a dict
- `d.keys()`, `d.values()`, `d.items()` (returning the view is atomic; see below for iterating it)
- `dict(d)` and `dict.fromkeys(iterable)` for well-known iterable types

Caveat: single-key operations call `__eq__` on keys, which can execute
arbitrary Python code and release the dict's lock mid-operation. For
built-in key types (`str`, `int`, `float`, `bytes`, `tuple` of same) this
is not a concern.

**Not thread-safe:**
- `for k in d: ...` / `for k, v in d.items(): ...` with any concurrent
  writer. Use `for k, v in d.copy().items(): ...` if you need to iterate
  a dict that other threads may mutate.
- `d[key] = d[key] + 1` (read-modify-write).
- `if key in d: del d[key]` (check-then-act). Use `d.pop(key, None)` or a
  `try/except KeyError`.

## `set`

Safe on shared sets:
- `len(s)`
- `s.add(elem)`, `s.remove(elem)`, `s.discard(elem)`, `s.pop()`
- `s.copy()`, `s.clear()`
- `elem in s` — lock-free, may observe concurrent modifications
- Binary operations (`s & other`, `s | other`, `s - other`, `s ^ other`)
  and in-place variants lock both operands when the other is a set,
  frozenset, or dict

Same `__eq__` caveat as dicts; not a concern for built-in element types.

**Not thread-safe:**
- `for elem in s: ...` with concurrent mutation — use `s.copy()` first.
- `if elem in s: s.remove(elem)` (check-then-act).

## `bytearray`

Safe on shared bytearrays: single-element and slice reads/writes, `append`,
`extend`, `insert`, `pop`, `remove`, `reverse`, `clear`, `copy`, and
methods like `find`, `replace`, `split`, `decode` (all hold the per-object
lock for their duration).

**Not thread-safe:** iteration under concurrent mutation; check-then-act
patterns.

## `memoryview`

- Creating/releasing a memoryview is thread-safe.
- Concurrent reads through memoryviews over an **immutable** source
  (`bytes`) are safe.
- Concurrent reads and/or writes through memoryviews over a **mutable**
  source (`bytearray`) are *not* safe and can corrupt the buffer. Even
  read-only memoryviews over a mutable source race against writers.

## Non-built-in containers and collections

The `collections` module types (`deque`, `defaultdict`, `OrderedDict`,
`Counter`, `ChainMap`) and `queue.Queue` / `queue.SimpleQueue` are *not*
covered by the built-in thread safety document. Treat them on a
case-by-case basis:

- `queue.Queue` and `queue.SimpleQueue` are explicitly designed to be
  thread-safe — `put` / `get` are safe.
- `collections.deque` — `append`, `appendleft`, `pop`, `popleft` are
  thread-safe by long-standing CPython guarantee (and remain so under
  free-threading), but iteration and compound operations are not.
- `collections.defaultdict` — inherits dict thread safety for direct
  operations, **but** `d[missing_key]` on a `defaultdict` triggers a
  factory call followed by a store, which is a compound read-modify-write
  operation and is *not* atomic. Auto-vivification on shared defaultdicts
  is a real race.
- `collections.Counter`, `OrderedDict`, `ChainMap` — assume not safe
  beyond the underlying dict guarantees.

## What this means for audits

1. **Do not report single `dict.__setitem__` / `list.append` / `set.add`
   on a shared container as corruption.** It isn't.
2. **Do report iteration under concurrent mutation.** The docs explicitly
   call this out as not thread-safe. The failure is silent (no crash, no
   exception), so reviewers may miss it without a prompt.
3. **Do report check-then-act / read-modify-write patterns** on shared
   containers — TOCTOU on `if key in d: del d[key]`, duplicate compilation
   from `if key not in cache: cache[key] = compile(...)`, and the classic
   `cache.get(k)` → conditional `cache[k] = v` are all real bugs.
4. **Do report defaultdict auto-vivification** on shared instances — this
   is a compound operation, not a single safe write.
5. **Classify the consequence concretely.** "The observed iteration may
   skip an element" is a different severity from "the container is
   corrupted". Free-threading rarely produces the latter for built-in
   containers; most issues are correctness, not memory safety.

