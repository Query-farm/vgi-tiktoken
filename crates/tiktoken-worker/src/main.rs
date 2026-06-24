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
                "tiktoken, tokens, token counting, tokenize, bpe, encoding, cl100k_base, \
                 o200k_base, llm, gpt-4, gpt-4o, context window, prompt budget, chunking, rag, \
                 truncate"
                    .to_string(),
            ),
            (
                "vgi.description_llm".to_string(),
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
                "vgi.description_md".to_string(),
                "# tiktoken\n\nExact LLM token counting and token-aware text chunking over Apache \
                 Arrow, powered by [`tiktoken-rs`](https://crates.io/crates/tiktoken-rs) (BPE \
                 encodings bundled — no network).\n\nScalars: `count_tokens`, `tokenize`, \
                 `truncate_to_tokens`, `chunk_by_tokens`, `encoding_for_model`, \
                 `tiktoken_version`.\n\nEncodings: cl100k_base, o200k_base, p50k_base, \
                 r50k_base, o200k_harmony."
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
                    "tiktoken, tokens, count_tokens, tokenize, encoding_for_model, \
                     truncate_to_tokens, chunk_by_tokens, bpe, cl100k_base, o200k_base, context \
                     window, rag chunking"
                        .to_string(),
                ),
                // VGI123 classifying tags (bare keys: domain/category/topic) for faceting.
                ("domain".to_string(), "llm".to_string()),
                ("category".to_string(), "tokenization".to_string()),
                ("topic".to_string(), "token-counting-and-chunking".to_string()),
                (
                    "vgi.source_url".to_string(),
                    "https://github.com/Query-farm/vgi-tiktoken/blob/main/crates/tiktoken-worker/src/main.rs"
                        .to_string(),
                ),
                (
                    "vgi.description_llm".to_string(),
                    "Token-aware text functions: count tokens, tokenize to BPE ids, truncate to a \
                     token budget, chunk into token-bounded windows, and map a model name to its \
                     tiktoken encoding."
                        .to_string(),
                ),
                (
                    "vgi.description_md".to_string(),
                    "LLM token counting and token-aware text chunking functions over Apache Arrow."
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
