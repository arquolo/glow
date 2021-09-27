# General

- Add docs for all exported functions

### `glow.__init__`

- Add explicit imports from glow.core.*

### `glow.cli.{parse_args -> run_cli}`

- Add support for arbitrary callable as first argument (with type hinted `__call__` method) to allow usage as decorator.

### `glow.mapped` (from `glow.core._parallel`)

Implement proper serialization of np.ndarray/np.memmap via anonymous mmap on Windows, and tmpfs mmap on Linux.

- `parent` -> `child` = move to shared memory if size allows
- `child` -> `parent` = keep in shared if already there, otherwise move as usual.
- Drop shared data at pool shutdown.

### `glow.{core.wrap -> core.cache}`

Add case `capacity=None` for unbound cache like in `functools`.

Use `evict: _Eviction` instead of `polycy` argument.

Combine all underlying modules to single module one, or find a better split.

Decorators for any callable with hashable args and kwargs:

- `call_once` - converts function to singleton (memoization of parameter-less function).
- `memoize` - cache calls with coalencing (unite with `shared_call`)

Decorators for callables accepting sequences of hashable items `(items: Sequence[Hashable]) -> list[Result]`:

- `stream_batched` - group calls to batches
- `memoize_batched` - cache and coalence calls

Improve test coverage.

### `glow.whereami`

- Improve function signature to show/hide stack frames from `site` modules.
  If 100% detection of foreign functions is not possible, skip only stdlib ones.

### `glow.core._len_helpers.{as_sized, partial_iter}`

- `len_hint(_object: Any) -> int: ...`
- Keep signature of wrapped function
- Make `len()` patching optional
- Add wrapper for `tqdm` to use there `len_hint(...)` instead of `total=len(...)`

### `glow.core._repr._Si`

Add proper string formatting using `format_spec`

### `glow.io._TiffImage`

- Enable fallthrough for bad tiles
- Use mmap and tile offsets from `libtiff` to decompose I/O from decoding to allow concurrent decoding.

### `glow.{nn.make_loader -?> utils.make_loader}`

- Seed as argument to toggle patching of dataset and iterable to provide batchsize- and workers-invariant data generation

### `glow.nn.auto`

- Drop module for use of `torch.nn.lazy.LazyModule`

### `glow.nn.modules`

- Drop garbage, redesign all of it
- Use glow.env as storage for options.

### `glow.{nn.plot -> utils.plot}`

- Fix plotting to collapse standard modules, instead of falling through into them.
- Refactor visitor to be more readable.
