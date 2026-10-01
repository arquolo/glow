# Changelog

## Unreleased

### In progress

- Extend the profiling wrapper to handle coroutines, awaitables, async iterators, and async generators, including suspend/resume tracking.
- Expand profiling tests for asynchronous execution, cancellation, exception propagation, and resource cleanup.

## 0.16.7

- Add `AsyncIterable` support to `chunked` and `windowed`, returning async iterators of tuples.
- Return one partial window from `windowed` when a nonempty source is shorter than the requested size, consistently across sequences, iterators, and async iterables.
- Reject `size < 1` in `windowed` and `chunked`. **Breaking change:** `windowed(..., 0)` now raises `ValueError` instead of returning an empty window.
- Fix slicing fallback for short objects such as `deque` that support integer indexing but not slices.
- Update public type annotations for async iterable inputs.
- Expand iterator tests to cover lazy consumption, partial results, source errors, cancellation, reference release, and invalid sizes.

## 0.16.6

- Fix cache overwrites and update cache size accounting.
- Improve dictionary compatibility of cache objects, including key iteration and expired-entry handling.

## 0.16.5

- Export `new_cache` for creating caches with item-count limits, byte limits, eviction policies, and TTL.
- Consolidate public type stubs in `glow/__init__.pyi` and add stubs for `glow.io`.
- Refine cache interfaces and type annotations across the package.

## 0.16.4

- Refactor batched cache dispatch and future-result handling.
- Refine exception handling in internal dispatch helpers.

## 0.16.3

- Add `memoize.drop` for invalidating cached entries through decorated functions, including batched and asynchronous functions.

## 0.16.2

- Fix a deadlock when cancelling work managed by `ThreadQuota`.
- Reduce redundant lock acquisition and release.
- Fix unobserved future exceptions in `memoize`.

## 0.16.1

- Fix `RwLock` synchronization and `MulticastQueue` cancellation handling.
- Fix iteration and termination in `azip`.
- Replace `ChainMap`-based environment handling with `contextvars`.
- Start the `streaming` worker pool on the first call.
- Remove the experimental wildcard export.

## 0.16.0

- Merge `astreaming` into `streaming`, which now supports both synchronous and asynchronous functions. **Breaking change:** replace calls to `astreaming` with `streaming`.
- Refactor caching and streaming internals.
- Support Python 3.13 and 3.14 (`>=3.13, <3.15`).

## 0.15.x

### Core and public API

- Add exports including `RwLock`, `MulticastQueue`, `afma`, `aminmax_norm`, `abs2`, `amap_dict`, `astreaming`, `cache_status`, `circle`, `clone_exc`, `declutter_tb`, `hide_frame`, `imresize`, `imresize_categorical`, `imrotate`, `maximum_cumsum`, `memtrack`, `span_task`, and `cumsum`.
- Standardize error messages and improve exception handling.

### Async execution and concurrency

- Redesign `memoize` with `count`, `nbytes`, `ttl`, and `policy` options (LRU, MRU, or no eviction policy).
- Allow item-count and byte limits to be used together.
- Improve thread safety, async function support, and batching.
- Add `astreaming`, an async counterpart to `streaming`, for batching requests by timeout or batch size.
- Support dynamic batch sizes in streaming.
- Consolidate legacy caching decorators around `memoize`.
- Add `MulticastQueue` for broadcasting items to async subscribers.

### Profiling and monitoring

- Introduce metric streams for tracking concurrent calls and busy/idle time.
- Add `memtrack` for periodically logging process RSS in the background.
- Limit the final profiling report to functions accounting for 95% of measured execution time.
- Add `memmon` for displaying process memory metrics (RSS, USS, PSS, and shared memory) in tables.

### Arrays and images

- Add `imresize`, `imresize_categorical`, `imrotate`, and `circle`.
- Add `afma` for array scaling and shifting, and `aminmax_norm` for lookup-table normalization of unsigned arrays.

### CLI, configuration, and typing

- Rework CLI parsing with `Meta` annotations, type inspection via `typing-inspection`, and module execution by name.
- Replace `cli.parse_args` with `cli.run`.
- Update Pydantic v2 integration through `__get_pydantic_json_schema__`.
- Improve generator and iterator wrapping, including `StopIteration`, `send`, and `throw` handling.
- Fix logging integration and Loguru stack levels.

### Dependencies and portability

- Add the `memmon` extra with `prettytable` and `psutil`.
- Add a GitHub Actions workflow for publishing tagged releases to PyPI using OIDC.
- Fix Windows-specific queue, future, thread-quota, and timeout issues.
- Improve shared-memory handling.

## 0.14.x

### Tests and maintenance

- Expand generator tests for `throw()`, `close()`, exception context, PEP 479 behavior, and frame-local cleanup after closing.
- Check that shared-memory buffers are initialized before writing to them in tests.
- Add `sizeof` coverage for primitives, nested collections, cyclic references, NumPy and PyTorch arrays, cached functions, and large-object performance.
- Address static-analysis warnings in thread-pool tests and update lint configuration.

## 0.13.x

### Architecture and breaking changes

- Remove PyTorch and deep-learning functionality, including `nn`, `metrics`, `transforms`, `distributed`, and the examples directory.
- Move utilities from `glow.core.*` into `glow.*`, focusing the library on general-purpose functional Python tools.

### New features

- Add integrated IceCream debugging through `ic` and `ic_repr`, with format specifications and NumPy array formatting for statistics, gradients, and packed bits.
- Add `aceil`, `afloor`, `apack`, `around`, and `pascal` array utilities.
- Add `groupby(iterable, key)`.
- Export `get_executor()`.
- Add `disable` to `timer()` and `time_this()`.
- Extend the `len()` patch to additional built-in iterators and collections through length hints.

### API and dependencies

- Improve SI-prefix formatting in `si` and `si_bin`, including automatic precision and format specifications.
- Add Pydantic 2.x validation and serialization support for `Uid`.
- Use `zip(..., strict=True)` in `map_n_dict` to validate result lengths.
- Remove obsolete compatibility wrappers and align code with Python 3.10+.
- Add the `ic` extra and replace `opencv-python` with `opencv-python-headless` in the `io` extra.

## 0.12.x

### Packaging

- Switch to Hatchling and `pyproject.toml`, removing `setup.py` and `setup.cfg`.
- Move package sources to `src/glow/`, with tests and examples at the repository root.

### Core utilities

- Improve `len()` support for built-in iterators such as `zip`, `map`, and `islice`.
- Rework `windowed`, `chunked`, and `ichunked`, including slicing support; add `ilen` and remove `as_sized` and `partial_iter`.
- Rewrite parallel execution, add `max_cpu_count()` with safeguards against Windows VMS leaks, and improve `buffered`.
- Add automatic chunk sizing, `map_n_dict`, and `ThreadQuota` for managing thread pools.
- Use nanosecond timing in `timer`; collect CPU/idle time and active-thread statistics in `time_this`, with a report at program exit.
- Support unlimited memoization caches (`capacity=None`) and byte-size limits (`bytesize`).
- Add `weak_memoize` and improve streaming batch handling.

### Neural networks

- Replace `make_loader` with the chainable `get_loader()` API, including `batch()`, `shuffle()`, and `pin_memory()`.
- Replace `Stepper` with `Trainer`, supporting train/eval stages, FP16/BF16 autocasting, and gradient accumulation through `grad_steps`.
- Replace `get_amp_context` with `get_grads`, supporting custom scalers, overflow retries, and learning-rate schedulers.
- Add `ConvCtx` for configuring convolutions, normalization, and activations.
- Add `DenseBlock`, `SqueezeExcitation`, `ResidualBlock`, `MaxVitBlock`, `VitBlock`, `Attention`, and `FeedForward`.
- Add `LazyLayerNorm`, `LazyGroupNorm`, and weight-standardized `Conv2dWs`.
- Add the `fc_densenet`, `vit`, and `max_vit` model factories.

### Metrics, I/O, CLI, and utilities

- Add sparse conversions (`to_index_sparse`, `to_prob_sparse`), improve DDP support, and add the Dice metric.
- Remove `io.Slide` and add `Sound.save()` with stronger typing.
- Improve dataclass CLI parsing for nested structures, `Optional`, and `Literal`, with fallback resolution of type annotations.
- Add `detach_()` and `eval_()` context managers and `materialize()` for initializing lazy modules.

### Breaking changes

- Remove `io.Slide`.
- Replace `make_loader` with `get_loader`, `Stepper` with `Trainer`, and `get_amp_context` with `get_grads`.

## 0.11.x

### Parallel execution

- Replace `mapped()` with `map_n()` and `starmap_n()`, with controls for `max_workers`, result order, prefetching, and chunk size.
- Rework `buffered` as a dataclass with improved queue handling and thread shutdown.

### CLI and images

- Extend `arg()` to recognize `Literal`, lists, and optional types, and support custom flags through `flag`.
- Add `prog` to `parse_args()` and improve positional and optional argument validation.
- Rename `TiledImage` and `read_tiled()` to `Slide` and `Slide.open()`.
- Unify multiscale SVS/TIFF image access with NumPy-style slicing, path-based caching, and improved handling of corrupted tiles.

### Transforms and profiling

- Rename `Compose` to `Chain`.
- Add probabilistic transform composition, `_Maybe` for conditional application, and `_OneOf` for random selection.
- Move `dither` into a separate module with Numba JIT optimization; improve `NumpyLike` typing and representations.
- Extend `time_this()` to profile generator iterations, reporting total time, time per call or iteration, execution percentage, and thread counts.

### PyTorch and dependencies

- Switch AMP to `torch.autocast` and inference contexts to `torch.inference_mode()`.
- Rename `num_workers` to `max_workers` in data loaders and parallel loaders.
- Fix state initialization in `SGDW`.
- Reorganize extras, including `cv-dither` and `memprof`, and update lint/type-check configuration.
- Simplify multiprocessing shared-memory transfer with `move_to_shmem()` and optimize memoization keys.

## 0.10.x

### New features

- Add `cli` for generating argument parsers from function signatures or dataclasses, supporting positional and named arguments, nested structures, and optional/list types.
- Add `api.env`, a `ChainMap`-based environment manager usable as a context manager or decorator, replacing `patch` and `Default`.
- Add `cv.Mosaic` for overlapping image tiles, parallel processing, and weighted merging to reduce boundary artifacts.
- Add `Uid`, a compact 22-character base57 UUID representation with serialization and deserialization.
- Add `nn.util.LazilyTraced` for lazy JIT tracing of PyTorch models with pickle support.

### API changes

- Rewrite `Sound`, removing `ObjectProxy`, adding `duration` and `channels`, and switching playback to `sounddevice` with Ctrl+C interruption support.
- Make `TiledImage.__getitem__` return a NumPy array directly; remove `view()` and improve slice-step, boundary, and background handling.
- Replace `Si` with `si()` and `si_bin()`, preserving numeric behavior while supporting formatted output.
- Allow `memprof` and `timer` to accept a callback or name and infer call locations through `whereami`.

### Architecture and performance

- Reorganize transforms into a package with `core`, `classes`, and `functional` modules, transform protocols, and dataclass decorators; remove the old flat API.
- Replace or remove legacy neural-network factories and provide building blocks for ResNet, ResNeSt, and WideResNet patterns, including `resblock`, `Cat`, `Sum`, `SEBlock`, `SplitAttention`, `conv`, and `upconv`.
- Remove `interpreter_lock` and forced thread-switching controls.
- Rewrite `sizeof` using CPython introspection (`_PyObject_GetDictPtr`) to return byte counts without recursive wrappers.
- Improve worker-shutdown error handling and submission queues in parallel pools.
- Improve alpha-channel and background-color handling in OpenSlide/TIFF image readers.

### Configuration and compatibility

- Consolidate development-tool configuration in `setup.cfg`.
- Reorganize installation extras as `glow[nn]`, `glow[io]`, `glow[cv]`, and `glow[all]`.
- Updating callers is required for the removal of `TiledImage.view()`, the replacement of `Si`, the transform package reorganization, and the removal of `interpreter_lock`.
