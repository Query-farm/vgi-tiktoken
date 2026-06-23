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
mod scalar;
mod tiktoken;

use vgi::Worker;

/// Worker version string, surfaced by `tiktoken_version()`.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
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

    let mut worker = Worker::new();
    scalar::register(&mut worker);
    worker.run();
}
