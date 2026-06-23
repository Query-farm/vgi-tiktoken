<p align="center">
  <img src="docs/vgi-logo.png" alt="Vector Gateway Interface (VGI)" width="320">
</p>

<p align="center"><em>A <a href="https://query.farm">Query.Farm</a> VGI worker for DuckDB.</em></p>

# vgi-tiktoken

A [VGI](https://query.farm) worker that brings **exact LLM token counting** and
**token-aware text chunking** to DuckDB over Apache Arrow, powered by
[`tiktoken-rs`](https://crates.io/crates/tiktoken-rs).

```sql
LOAD vgi;
ATTACH 'tiktoken' (TYPE vgi, LOCATION './target/release/tiktoken-worker');
SET search_path = 'tiktoken.main';

SELECT count_tokens('hello world');                      -- 2   (cl100k_base default)
SELECT count_tokens('hello world', 'gpt-4o');             -- 2   (o200k_base)
SELECT tokenize('tiktoken is great!');                    -- [83, 1609, 5963, 374, 2294, 0]
SELECT truncate_to_tokens('a long passage …', 16);        -- first 16 tokens, decoded
SELECT chunk_by_tokens('… RAG document …', 256, 32);      -- VARCHAR[] windows w/ overlap
SELECT encoding_for_model('gpt-4o');                      -- 'o200k_base'
SELECT tiktoken_version();                                -- worker version
```

## No network, exact BPE

`tiktoken-rs` **bundles** the OpenAI byte-pair encodings (cl100k_base,
o200k_base, p50k_base, r50k_base, o200k_harmony) directly into the binary — there
is **no model download** at runtime. Each encoder is built lazily on first use
and cached for the process lifetime, so the first call to an encoding pays the
build cost and every subsequent call is free.

Counts are **exact** for the OpenAI BPE families (GPT-4 / 4o / 4.1 / 5, the
o-series, GPT-3.5-turbo, and the legacy davinci/codex models). For **other**
model families (Anthropic Claude, Meta Llama, Mistral, Google Gemini, …) the
result is a **close proxy** but not exact — those vendors ship different
tokenizers. Use it as an estimate there.

## Function surface

Scalars (positional-only; optional arguments are exposed as **arity overloads**):

| Function | Signature | Notes |
| --- | --- | --- |
| `count_tokens` | `count_tokens(text VARCHAR) -> INTEGER` | Default encoding cl100k_base; empty → 0 |
| `count_tokens` | `count_tokens(text VARCHAR, model VARCHAR) -> INTEGER` | Encoding chosen by model name; unknown model → **NULL** |
| `tokenize` | `tokenize(text VARCHAR) -> INTEGER[]` | Token ids, default cl100k_base |
| `tokenize` | `tokenize(text VARCHAR, model VARCHAR) -> INTEGER[]` | Token ids for `model`'s encoding |
| `truncate_to_tokens` | `truncate_to_tokens(text VARCHAR, n INTEGER) -> VARCHAR` | First `n` tokens, decoded back to text |
| `truncate_to_tokens` | `truncate_to_tokens(text VARCHAR, n INTEGER, model VARCHAR) -> VARCHAR` | Same, with `model`'s encoding |
| `chunk_by_tokens` | `chunk_by_tokens(text VARCHAR, max_tokens INTEGER) -> VARCHAR[]` | Split into ≤ `max_tokens` chunks (no overlap) |
| `chunk_by_tokens` | `chunk_by_tokens(text VARCHAR, max_tokens INTEGER, overlap INTEGER) -> VARCHAR[]` | RAG windows with `overlap` shared tokens |
| `encoding_for_model` | `encoding_for_model(model VARCHAR) -> VARCHAR` | Encoding name (e.g. `'o200k_base'`); **NULL** if unknown |
| `tiktoken_version` | `tiktoken_version() -> VARCHAR` | Worker version |

### Encodings supported

| Encoding | Model families |
| --- | --- |
| `cl100k_base` | GPT-4, GPT-4-turbo, GPT-3.5-turbo, text-embedding-3/ada-002 |
| `o200k_base` | GPT-4o, GPT-4.1, GPT-4.5, the o-series (o1/o3/o4) |
| `p50k_base` | text-davinci-002/003, the codex models |
| `r50k_base` (gpt2) | davinci/curie/babbage/ada (legacy) |
| `o200k_harmony` | gpt-oss harmony format |

You may pass either a **model name** (`'gpt-4o'`) or a **canonical encoding name**
(`'o200k_base'`) wherever a `model` argument is accepted.

### NULL / error / empty policy

- **Unknown model** → **NULL** (treated as missing metadata, never an error, so a
  dirty model column doesn't abort the scan). Find them with
  `WHERE encoding_for_model(model) IS NOT NULL`.
- **NULL input** → **NULL** output (a NULL list for list-returning functions).
- **Empty text** → `0` for `count_tokens`, `[]` for `tokenize` / `chunk_by_tokens`,
  `''` for `truncate_to_tokens`.
- `max_tokens <= 0` for `chunk_by_tokens` → empty list. `overlap` is clamped to
  `max_tokens - 1` so the window always advances.

## Chunking model

`chunk_by_tokens` tokenizes once, then slides a window of `max_tokens` tokens
forward by `max_tokens - overlap` each step, decoding each window back to a valid
string. Concatenating the **no-overlap** chunks reconstructs the input exactly;
with overlap, dropping the leading `overlap` tokens of each subsequent chunk
reconstructs the original token stream. Decoding uses a lossy UTF-8 conversion so
a window that splits a multi-byte character still yields a valid `String`.

## Development

```sh
make test       # cargo unit/integration tests + SQL E2E
make test-unit  # cargo test --workspace
make test-sql   # build release worker + DuckDB sqllogictest suite (haybarn-unittest)
make lint       # clippy (deny warnings) + rustfmt --check
make fmt        # rustfmt the workspace
```

The SQL E2E suite uses [`haybarn-unittest`](https://query.farm)
(`uv tool install haybarn-unittest`).

## License

MIT — see [LICENSE](LICENSE). `tiktoken-rs` is MIT-licensed.

---

## Authorship & License

Written by [Query.Farm](https://query.farm) — every VGI worker is designed and built by Query.Farm.

Copyright 2026 Query Farm LLC - https://query.farm

