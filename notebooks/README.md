# Token-counting notebooks

[01 · One character budget is not a token budget](01_characters_bytes_and_tokens.ipynb)
compares Unicode code points, UTF-8 bytes and exact ordinary-text BPE token
counts for eleven fixed strings. Visible SQL calls the real released worker,
checks token-list lengths and NULL behavior, and contrasts two encodings. Two
embedded charts show the counts and selected encoding's ratios. A final boundary
control demonstrates why a one-token cut can change Unicode text.

## Run from this directory

Install [uv](https://docs.astral.sh/uv/), then:

```sh
cd notebooks
uv sync --locked
uv run --locked python execute.py
```

In a notebook editor select this directory's `.venv/bin/python` kernel and run
all cells. The complete notebook directory also works outside the worker checkout.
Python 3.13–3.14 is supported; `uv` can provision the interpreter. No Rust compiler,
local worker build, API key or root virtual environment is required.

The notebook displays its literal connection:

```sql
INSTALL vgi FROM community;
LOAD vgi;
ATTACH 'tokens' (TYPE vgi,
  LOCATION 'github-auto://Query-farm/vgi-tiktoken@v0.4.1');
```

VGI selects the native platform archive, verifies its SHA256 sidecar and caches
it. The release supplies macOS arm64 and Linux amd64/arm64 assets. Initial
installation requires network access; the worker bundles its encoding
vocabularies and tokenizes local text. Haybarn 2.0.0rc11 and the notebook packages
are locked here. To adopt another release, change the visible tag and rerun both
the default and experiment before relying on saved results.

Parameters, connection SQL and analytical SQL stay expanded. Imports, plotting,
prose-generation source and cleanup are folded, with useful results visible.

## One-parameter experiment

```sh
uv run --locked python execute.py --parameters '{"ENCODING":"o200k_base"}' \
  --output-dir validation/o200k
```

Accepted values are `cl100k_base` (default) and `o200k_base` (experiment). Both
encodings appear in the fixed reference comparison. The parameter selects the
bar outline, detailed table and ratios. The source text, byte/code-point lengths,
full reference comparison and explicit cl100k_base boundary control stay fixed.
Reset to `cl100k_base` after comparing results.

The runner uses fresh kernels and saves only successful executions. Parameter
overrides require a separate destination; unknown parameter names and overwriting
the source directory are rejected. Copies include assets, helpers, this guide and
the locked environment. Extracted charts, logs and experiments under `validation/`
are ignored; the main notebook retains successful outputs and embedded charts.

## Provenance and limits

All eleven strings are original teaching examples embedded in SQL: short
sentences, JSON, an identifier, digits, indentation and two accent representations.
They are neither production prompts nor a representative multilingual benchmark.
No API is called, and no account data or credentials are stored.

`length` counts Unicode code points, not visible grapheme clusters. UTF-8 bytes
are measured with `octet_length(encode(text))`. No normalization is performed;
precomposed and combining accents remain distinct byte sequences. Empty strings
produce zero tokens, while missing text or unknown encoding names produce NULL.

The released worker uses ordinary-text BPE encoding. Canonical encoding names are
passed explicitly rather than relying on model aliases. Counts describe these
strings only, excluding chat serialization, tool definitions, images and other
application overhead. They do not establish API usage, pricing, model quality or
counts for models with different tokenizers. Token IDs belong to their encoding.

The v0.4.1 truncation and chunking implementations decode byte fragments with
lossy UTF-8 conversion. A cut inside a character can yield the replacement
character U+FFFD. The notebook demonstrates this with a cl100k_base emoji boundary;
it does not promise that concatenating decoded chunks reconstructs arbitrary
Unicode input. The selected comparison encoding does not alter this explicit
boundary control.

The official Strata Sun and VGI media-kit assets are unchanged. Linked Strata Sun
branding appears at both the start and final end of the notebook.

References: [worker documentation](https://github.com/Query-farm/vgi-tiktoken),
[pinned release](https://github.com/Query-farm/vgi-tiktoken/releases/tag/v0.4.1),
[tokenization implementation](https://github.com/Query-farm/vgi-tiktoken/blob/v0.4.1/crates/tiktoken-worker/src/tiktoken.rs).
