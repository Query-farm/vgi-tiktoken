//! The `tiktoken` VGI worker.
//!
//! A standalone binary that DuckDB launches and talks to over Apache Arrow IPC
//! (`ATTACH 'tiktoken' (TYPE vgi, LOCATION '…')`). It brings exact LLM token
//! counting and token-aware text chunking to SQL under the catalog `tiktoken`,
//! schema `main`:
//!
//! ```sql
//! ATTACH 'tiktoken' (TYPE vgi, LOCATION './target/release/tiktoken-worker');
//! SET search_path = 'tiktoken.main';
//!
//! SELECT count_tokens('hello world');               -- 2  (cl100k_base default)
//! SELECT count_tokens('hello world', 'gpt-4o');      -- 2  (o200k_base)
//! SELECT tokenize('tiktoken is great!', 'gpt-4');    -- [83, 1609, ...]  INT[]
//! SELECT truncate_to_tokens('a long passage …', 16); -- first 16 tokens, decoded
//! SELECT chunk_by_tokens('… RAG document …', 256, 32); -- VARCHAR[] windows
//! SELECT encoding_for_model('gpt-4o');               -- 'o200k_base'
//! SELECT tiktoken_version();                         -- worker version
//! ```
//!
//! The pure tokenization engine (wrapping `tiktoken-rs`, which bundles the BPE
//! encodings — no network) lives in `tiktoken.rs`; the `scalar/` modules are
//! thin Arrow adapters over it.

mod arrow_io;
mod meta;
mod scalar;
mod tiktoken;

use vgi::catalog::{CatSchema, CatalogModel};
use vgi::Worker;

/// Worker version string, surfaced by `tiktoken_version()`.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Catalog + schema metadata (description, provenance) surfaced to DuckDB and
/// the `vgi-lint` metadata-quality linter. The function objects themselves are
/// served from the registered scalars; this only adds catalog/schema-level
/// comments and tags.
fn catalog_metadata(name: &str) -> CatalogModel {
    CatalogModel {
        name: name.to_string(),
        comment: Some(
            "Exact LLM token counting and token-aware text chunking, powered by tiktoken-rs."
                .to_string(),
        ),
        tags: vec![
            (
                "vgi.title".to_string(),
                "Tiktoken — LLM Token Counting & Chunking".to_string(),
            ),
            (
                "vgi.keywords".to_string(),
                meta::keywords_json(&[
                    "tiktoken",
                    "tokens",
                    "token counting",
                    "tokenize",
                    "bpe",
                    "encoding",
                    "cl100k_base",
                    "o200k_base",
                    "llm",
                    "gpt-4",
                    "gpt-4o",
                    "context window",
                    "prompt budget",
                    "chunking",
                    "rag",
                    "truncate",
                ]),
            ),
            (
                "vgi.doc_llm".to_string(),
                "Count exact LLM tokens for text, tokenize text to BPE token ids, map a model \
                 name to its tiktoken encoding, truncate text to a token budget, and split text \
                 into token-bounded (optionally overlapping) chunks for RAG. Encodings are \
                 OpenAI's BPE families (cl100k_base for GPT-4/3.5, o200k_base for GPT-4o, plus \
                 p50k_base / r50k_base / o200k_harmony) and are bundled into the worker — no \
                 network access. Use to budget prompts/context windows, estimate API token cost, \
                 and chunk documents before embedding."
                    .to_string(),
            ),
            (
                "vgi.doc_md".to_string(),
                "# tiktoken — Count LLM Tokens & Chunk Text in SQL\n\n\
                 ![tiktoken logo](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/OpenAI_Logo.svg/250px-OpenAI_Logo.svg.png)\n\n\
                 **Count LLM tokens, tokenize text, and split documents into token-bounded chunks \
                 directly in DuckDB SQL** — exact OpenAI-compatible BPE token counting and \
                 token-aware text chunking for prompt budgeting, API cost estimation, and \
                 retrieval-augmented generation (RAG), with no Python and no network calls.\n\n\
                 This extension is for anyone who needs to know exactly how many tokens a piece of \
                 text will consume before sending it to a large language model. Instead of \
                 approximating with character or word counts, `tiktoken` computes the *real* token \
                 count using the same byte-pair-encoding (BPE) vocabularies that OpenAI's GPT models \
                 use. That makes it ideal for staying within context-window limits, estimating the \
                 cost of an API call, deduplicating or filtering rows by token length, and preparing \
                 large corpora for embedding pipelines — all from ordinary SQL over your existing \
                 tables.\n\n\
                 Under the hood the worker wraps the Rust crate \
                 [`tiktoken-rs`](https://github.com/zurawiki/tiktoken-rs) \
                 ([docs](https://docs.rs/tiktoken-rs)), a fast, pure-Rust port of OpenAI's \
                 [`tiktoken`](https://github.com/openai/tiktoken) tokenizer. The BPE encodings are \
                 **bundled into the binary** and built lazily on first use, so token counting works \
                 fully offline with no model download. The supported encodings cover every modern \
                 OpenAI model family: `cl100k_base` (GPT-4 / GPT-3.5), `o200k_base` (GPT-4o), \
                 `p50k_base`, `r50k_base`, and `o200k_harmony`. Results are exact and deterministic, \
                 so the same text always yields the same count and the same token-id sequence.\n\n\
                 The worker exposes a small, composable set of scalar SQL functions in the \
                 `tiktoken.main` schema; list that schema to discover them and their signatures. \
                 They cover the everyday token-oriented tasks: getting an exact token count for a \
                 string, returning the raw BPE token ids as an `INTEGER[]`, mapping a model name \
                 (e.g. `'gpt-4o'`) to its encoding name, clipping text to a fixed token budget, and \
                 splitting text into token-bounded, optionally overlapping windows that make ideal \
                 RAG chunks before embedding. Each text function offers a default-encoding form \
                 (`cl100k_base`) and a model-aware form that accepts a model or encoding name; \
                 unknown model names return `NULL`, and `NULL` input flows through to `NULL`. Built \
                 and maintained by [Query.Farm](https://query.farm)."
                    .to_string(),
            ),
            ("vgi.author".to_string(), "Query.Farm".to_string()),
            (
                "vgi.copyright".to_string(),
                "Copyright 2026 Query Farm LLC - https://query.farm".to_string(),
            ),
            ("vgi.license".to_string(), "MIT".to_string()),
            (
                "vgi.support_contact".to_string(),
                "https://github.com/Query-farm/vgi-tiktoken/issues".to_string(),
            ),
            (
                "vgi.support_policy_url".to_string(),
                "https://github.com/Query-farm/vgi-tiktoken/blob/main/README.md".to_string(),
            ),
            // VGI152: an analyst-task suite so `vgi-lint simulate` can measure how
            // well an agent actually drives this worker. Each task is a natural
            // prompt plus the canonical `reference_sql` that answers it. Every task
            // is deterministic and self-contained (tiktoken is exact and bundles
            // its encodings), so none needs external fixtures.
            (
                "vgi.agent_test_tasks".to_string(),
                r#"[
  {"name": "worker_version", "prompt": "Which version of the tiktoken worker is currently attached? Return the single version string in a column named worker_version.", "reference_sql": "SELECT tiktoken.main.tiktoken_version() AS worker_version"},
  {"name": "count_default_encoding", "prompt": "How many GPT-4 / GPT-3.5 (cl100k_base) tokens are in the text 'hello world'? Return the count in a column named token_count.", "reference_sql": "SELECT tiktoken.main.count_tokens('hello world') AS token_count"},
  {"name": "count_for_model", "prompt": "Using the tokenizer for the model 'gpt-4o', how many tokens are in the text 'hello world'? Return the count in a column named token_count.", "reference_sql": "SELECT tiktoken.main.count_tokens('hello world', 'gpt-4o') AS token_count"},
  {"name": "encoding_for_model", "prompt": "Which tiktoken encoding does the model 'gpt-4o' use? Return the encoding name in a column named encoding.", "reference_sql": "SELECT tiktoken.main.encoding_for_model('gpt-4o') AS encoding"},
  {"name": "tokenize_ids", "prompt": "Return the raw cl100k_base BPE token ids for the text 'hello world' as an integer array in a column named token_ids.", "reference_sql": "SELECT tiktoken.main.tokenize('hello world') AS token_ids"},
  {"name": "truncate_to_budget", "prompt": "Truncate the text 'The quick brown fox jumps over the lazy dog.' to its first 5 cl100k_base tokens and return the resulting string in a column named clipped.", "reference_sql": "SELECT tiktoken.main.truncate_to_tokens('The quick brown fox jumps over the lazy dog.', 5) AS clipped"},
  {"name": "chunk_count", "prompt": "Split the text 'The quick brown fox jumps over the lazy dog.' into chunks of at most 5 tokens (cl100k_base) and return how many chunks result, in a column named chunk_count.", "reference_sql": "SELECT len(tiktoken.main.chunk_by_tokens('The quick brown fox jumps over the lazy dog.', 5)) AS chunk_count"}
]"#
                .to_string(),
            ),
        ],
        source_url: Some("https://github.com/Query-farm/vgi-tiktoken".to_string()),
        schemas: vec![CatSchema {
            name: "main".to_string(),
            comment: Some(
                "LLM token counting and token-aware text chunking functions.".to_string(),
            ),
            tags: vec![
                ("vgi.title".to_string(), "Tiktoken — main".to_string()),
                (
                    "vgi.keywords".to_string(),
                    meta::keywords_json(&[
                        "tiktoken",
                        "tokens",
                        "count_tokens",
                        "tokenize",
                        "encoding_for_model",
                        "truncate_to_tokens",
                        "chunk_by_tokens",
                        "bpe",
                        "cl100k_base",
                        "o200k_base",
                        "context window",
                        "rag chunking",
                    ]),
                ),
                // VGI123 classifying tags (bare keys: domain/category/topic) for faceting.
                ("domain".to_string(), "llm".to_string()),
                ("category".to_string(), "tokenization".to_string()),
                ("topic".to_string(), "token-counting-and-chunking".to_string()),
                // VGI413: the schema's category registry — an ordered list of the
                // navigation sections its objects are grouped into. Each function
                // carries a `vgi.category` naming one of these.
                (
                    "vgi.categories".to_string(),
                    r#"[
  {"name": "count", "description": "Count the exact number of LLM tokens in text."},
  {"name": "tokenize", "description": "Convert text into its raw BPE token ids."},
  {"name": "shape", "description": "Fit text to a token budget by truncating it or splitting it into token-bounded chunks for RAG."},
  {"name": "reference", "description": "Map model names to tiktoken encodings and inspect the worker build."}
]"#
                    .to_string(),
                ),
                // VGI139: per-object source_url is omitted; provenance lives only
                // on the catalog object (see `source_url` above).
                (
                    "vgi.doc_llm".to_string(),
                    "Token-aware text functions over Apache Arrow. They compute exact LLM token \
                     counts, expose the raw BPE token ids of a string, clip text to a fixed token \
                     budget, split text into token-bounded (optionally overlapping) RAG windows, \
                     and map model names to their tiktoken encodings. Each text function has a \
                     default-encoding form (cl100k_base) and a model-arity form that resolves a \
                     model or encoding name. Unknown models return NULL; NULL text flows through \
                     to NULL. List the schema to discover the exact functions and their signatures."
                        .to_string(),
                ),
                (
                    "vgi.doc_md".to_string(),
                    "## tiktoken.main\n\nThe primary schema of the tiktoken worker. It brings \
                     **exact LLM token counting** and **token-aware text chunking** to DuckDB SQL \
                     over Apache Arrow, grouped into a few areas: **counting** tokens, \
                     **tokenizing** text into raw BPE ids, **shaping** text to fit a token budget, \
                     and **reference** lookups that map a model name to its encoding and report \
                     the worker build.\n\nThe default encoding is `cl100k_base` (GPT-4 / GPT-3.5); \
                     pass a model name (e.g. `gpt-4o`) to select another. Encodings are bundled \
                     into the binary, so token counting is exact, deterministic, and needs no \
                     network access. List the schema to discover the exact functions and their \
                     signatures."
                        .to_string(),
                ),
                // VGI506 representative example queries for the schema.
                (
                    "vgi.example_queries".to_string(),
                    "SELECT tiktoken.main.count_tokens('The quick brown fox jumps over the lazy dog.');\n\
                     SELECT tiktoken.main.count_tokens('Summarize this prompt.', 'gpt-4o');\n\
                     SELECT tiktoken.main.tokenize('tiktoken is great!');\n\
                     SELECT tiktoken.main.encoding_for_model('gpt-4o');\n\
                     SELECT tiktoken.main.truncate_to_tokens('The quick brown fox jumps over the lazy dog.', 5);\n\
                     SELECT tiktoken.main.chunk_by_tokens('A long document to split before embedding.', 8);"
                        .to_string(),
                ),
            ],
            views: Vec::new(),
            macros: Vec::new(),
            tables: Vec::new(),
        }],
        ..Default::default()
    }
}

fn main() {
    // Logs MUST go to stderr — stdout is the Arrow-IPC channel.
    let _ = env_logger::Builder::from_env(env_logger::Env::default().filter_or("VGI_LOG", "info"))
        .format_timestamp_millis()
        .try_init();

    // The catalog name DuckDB sees in `ATTACH 'tiktoken' (TYPE vgi, …)`. Default
    // to `tiktoken`, but honor an explicit override so a test harness can rename
    // it.
    if std::env::var_os("VGI_WORKER_CATALOG_NAME").is_none() {
        std::env::set_var("VGI_WORKER_CATALOG_NAME", "tiktoken");
    }
    let catalog_name =
        std::env::var("VGI_WORKER_CATALOG_NAME").unwrap_or_else(|_| "tiktoken".to_string());

    let mut worker = Worker::new();
    scalar::register(&mut worker);
    worker.set_catalog(catalog_metadata(&catalog_name));
    worker.run();
}
