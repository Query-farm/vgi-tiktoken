# CLAUDE.md — vgi-tiktoken

Contributor/agent notes. User-facing docs live in `README.md`; this is the
"how it's built and where the sharp edges are" companion.

## What this is

A [VGI](https://query.farm) worker (Rust, compiled binary) exposing **exact LLM
token counting** and **token-aware text chunking** to DuckDB/SQL over Arrow IPC.
Built on the `vgi` crate (crates.io), modeled on `vgi-units` / `vgi-color` /
`vgi-ioc`. Catalog name `tiktoken` (single `main` schema).

The tokenization engine wraps [`tiktoken-rs`](https://crates.io/crates/tiktoken-rs),
which **bundles** the OpenAI BPE encodings into the binary — **no network
download** at runtime. Encoders are built lazily and cached for the process
lifetime via `once_cell`.

## Layout

```
Cargo.toml                          workspace; pins vgi = "0.5.0", arrow 58
crates/tiktoken-worker/
  src/main.rs                       Worker::new(); registers scalars
  src/lib.rs                        lib target re-exporting `tiktoken` for integration tests
  src/tiktoken.rs                   PURE engine (no Arrow): encoding map + count/tokenize/decode/truncate/chunk + unit tests
  src/arrow_io.rs                   VARCHAR/INT cell reads + LIST(INT)/LIST(VARCHAR) builders + in-process scalar test harness
  src/scalar/{count,tokenize,truncate,chunk,encoding,version,mod}.rs   thin Arrow scalar adapters
  tests/tokenization.rs             integration tests against KNOWN token counts/ids
test/sql/*.test                     haybarn-unittest sqllogictest — authoritative E2E
Makefile                            test / test-unit / test-sql / lint / fmt / build / clean
```

Pattern: keep computation in `tiktoken.rs` (pure, unit-tested), keep Arrow
marshalling in `arrow_io.rs` + `scalar/*.rs` (thin, harness-tested).

## Sharp edges

1. **`tiktoken-rs` is PINNED to `=0.11.0`.** 0.12 uses `let`-chains in `&&`
   position, which are stable only in Rust **1.88**; our workspace MSRV is
   **1.86**, so 0.12 fails to build (`E0658: let expressions … are unstable`).
   0.11.0 has MSRV 1.73 and the same API surface we use
   (`*_singleton() -> &'static CoreBPE`, `encode_ordinary`, `decode_bytes`,
   `bpe_for_tokenizer`, `tokenizer::get_tokenizer`). **Do not bump to 0.12 until
   the workspace MSRV reaches ≥ 1.88.** Note `Tokenizer` lives at
   `tiktoken_rs::tokenizer::Tokenizer` (not re-exported at the crate root in
   0.11) and has a `Gpt2` variant the model→encoding match must handle.

2. **`haybarn-unittest` skips `require vgi`** — `.test` files use explicit
   `statement ok` + `LOAD vgi;`. Functions live under the `tiktoken` catalog, so
   each file does `SET search_path = 'tiktoken.main'`, then `USE memory` before
   `DETACH tiktoken`. Determinism: tiktoken is exact, so token counts and id
   sequences are asserted literally (no rounding needed).

3. **Optional args = arity overloads.** `count_tokens`, `tokenize`,
   `truncate_to_tokens`, and `chunk_by_tokens` each register **two** (or three)
   `ScalarFunction` impls that share a name; VGI's overload resolver
   (`overload.rs`) picks by argument count. The default-arity impls use
   `Encoding::default_encoding()` (cl100k_base); the model-arity impls call
   `tiktoken::resolve(model)`, which accepts a model name **or** a canonical
   encoding name.

4. **Unknown-model policy (deliberate).** An unknown model string → **NULL**
   (consistent with `encoding_for_model`), not an error. This treats a dirty
   model column as missing metadata rather than aborting a scan.

5. **LIST return types must match bind↔process.** `tokenize` returns
   `LIST(INTEGER)` and `chunk_by_tokens` returns `LIST(VARCHAR)`. The element
   field is named `item`/nullable, produced once by `arrow_io::list_int_type()` /
   `list_varchar_type()` and used in both `on_bind` and `process`. NULL in → NULL
   list; empty/no-result → empty list.

6. **Chunking terminates and covers.** `tiktoken::chunk` clamps `overlap` to
   `max_tokens - 1` so the window always advances; `max_tokens == 0` → empty.
   Decoding is lossy UTF-8 so a window splitting a multi-byte char still yields a
   valid string. Tested: each chunk ≤ max, and re-tokenizing the chunks (dropping
   the repeated overlap) reconstructs the original token stream exactly.

7. **Token ids are `u32` (`Rank`) → `INT32`.** We narrow to `i32` for the Arrow
   `LIST(INT)`; real BPE vocabularies (≤ ~200k) fit comfortably.

## Testing

```sh
cargo test --workspace --all-features    # pure unit + arrow-boundary harness + integration
cargo clippy --all-targets --all-features -- -D warnings && cargo fmt --all -- --check
make test-sql                            # builds release, sets VGI_TIKTOKEN_WORKER, haybarn over test/sql/*
make test                                # cargo test + sql
```

KNOWN constants used in tests (deterministic): cl100k_base `count('hello world')`
= 2 (ids `[15339, 1917]`); `'tiktoken is great!'` = 6 ids
`[83, 1609, 5963, 374, 2294, 0]`; o200k_base `'hello world'` ids `[24912, 2375]`;
`encoding_for_model('gpt-4o')` = `o200k_base`.

CI (`.github/workflows/ci.yml`) runs fmt/clippy/build/test plus a gated
`e2e-sql` job (installs `uv` + `haybarn-unittest`, runs `make test-sql`).

## Function surface

Scalars: `count_tokens` (INT, 1- or 2-arg), `tokenize` (INT[], 1- or 2-arg),
`truncate_to_tokens` (VARCHAR, 2- or 3-arg), `chunk_by_tokens` (VARCHAR[], 2- or
3-arg), `encoding_for_model` (VARCHAR), `tiktoken_version` (VARCHAR). Encodings:
cl100k_base, o200k_base, p50k_base, r50k_base, o200k_harmony.
