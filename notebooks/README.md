# Token counts and overlapping windows

Two executed SQL narratives use the native [v0.4.1 worker release](https://github.com/Query-farm/vgi-tiktoken/releases/tag/v0.4.1):

1. [Characters, bytes and tokens](01_characters_bytes_and_tokens.ipynb) compares eleven original strings under two encodings, resolves stored model labels, inspects token IDs and demonstrates a lossy truncation boundary.
2. [Overlap repeats tokens](02_overlap_repeats_tokens_and_boundaries_matter.ipynb) reconciles returned windows against the original token IDs, measures repeated context and shows why decoded chunks cannot always reconstruct Unicode text.

Each notebook has an **Open in Colab** button. Open a fresh runtime and run all cells; only the shared base package is installed. No repository checkout, Rust compiler or API key is required. The worker connection remains visible:

```sql
ATTACH 'tokens' (TYPE vgi,
  LOCATION 'github-auto://Query-farm/vgi-tiktoken@v0.4.1');
```

Initial setup downloads the platform release and checksum-verified plotting companion. Tokenization then uses the worker's bundled vocabularies and local text. GitHub displays the saved SQL, results and SVG charts. It does not honor Jupyter cell-folding metadata, so setup is kept short; compatible notebook editors may collapse auxiliary cells.

## Run locally

```sh
cd notebooks
uv sync --locked
uv run --locked vgi-notebooks --timeout 300
```

The local environment includes the shared runner extra. Colab installs the base package, preserving compatible preloaded pandas, Matplotlib and kernel packages. Presentation, execution and the fleet standard are maintained in [vgi-example-notebooks](https://github.com/Query-farm/vgi-example-notebooks); this repo keeps its narratives, chart companion and auditable coverage evidence.

## Change one parameter

```sh
uv run --locked vgi-notebooks 01_characters_bytes_and_tokens.ipynb \
  --parameters '{"ENCODING":"o200k_base"}' --output-dir artifacts/o200k
uv run --locked vgi-notebooks 02_overlap_repeats_tokens_and_boundaries_matter.ipynb \
  --parameters '{"OVERLAP_TOKENS":0}' --output-dir artifacts/zero-overlap
```

The first experiment changes the selected encoding's ratios and bar outline while retaining the same strings and two-encoding comparison. The second changes three overlapping windows totaling twenty tokens into two disjoint windows totaling sixteen. The explicit cl100k_base Unicode controls stay fixed.

## Scope and evidence

All inputs are original teaching strings embedded in visible SQL. They are not a language benchmark, real prompts or complete API usage estimates. No model or embedding service is called. Code points, UTF-8 bytes, individual-window token counts and concatenated-text counts measure different things. Model names resolve according to the pinned worker's bundled mapping.

The released worker uses lossy UTF-8 decoding for truncation and chunking. The examples preserve original IDs and inspect replacement characters; they do not promise lossless reconstruction of arbitrary Unicode input. Chunking uses cl100k_base and has no encoding argument.

[coverage.json](coverage.json) links the executed examples to all five logical public functions (nine overloads), with complete function/table/view exports and source evidence under [coverage](coverage/). Current-main mappings are reviewed for interface compatibility; notebook runtime evidence is explicitly the released worker. See the [shared coverage rules](https://github.com/Query-farm/vgi-example-notebooks/blob/main/docs/coverage.md).
