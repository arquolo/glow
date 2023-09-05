# General

- Add docs for all exported functions

### `__init__`

### `{map,starmap}_n` (from `._parallel`)

Implement proper serialization of np.ndarray/np.memmap via anonymous mmap on Windows, and tmpfs mmap on Linux.

- `parent` -> `child` = move to shared memory if size allows
- `child` -> `parent` = keep in shared if already there, otherwise move as usual.
- Drop shared data at pool shutdown.

### `.wrap -> .cache`

Add case `capacity=None` for unbound cache like in `functools`.

Use `evict: _Eviction` instead of `policy` argument.

Combine all underlying modules to single module one, or find a better split.

Decorators for any callable with hashable args and kwargs:

- `call_once` - converts function to singleton (memoization of parameter-less function).
- `memoize` - cache calls with coalencing (unite with `shared_call`)

Decorators for callables accepting sequences of hashable items `(items: Sequence[Hashable]) -> list[Result]`:

- `stream_batched` - group calls to batches
- `memoize_batched` - cache and coalence calls

Improve test coverage.

### `whereami`

- Improve function signature to show/hide stack frames from `site` modules.
  If 100% detection of foreign functions is not possible, skip only stdlib ones.

### `._patch_len`

- `len_hint(_object: Any) -> int: ...`
- Keep signature of wrapped function
- Make `len()` patching optional
- Add wrapper for `tqdm` to use there `len_hint(...)` instead of `total=len(...)`

### `._repr._Si`

Add proper string formatting using `format_spec`

## `io` contents
