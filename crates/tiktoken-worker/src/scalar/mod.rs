//! Scalar functions exposed by the tiktoken worker, registered under
//! `tiktoken.main`.
//!
//! Functions with optional arguments are exposed as **arity overloads**: two (or
//! three) `ScalarFunction` impls share a name, and VGI's overload resolver picks
//! the one whose argument count matches the call.

mod chunk;
mod count;
mod encoding;
mod tokenize;
mod truncate;

use vgi::Worker;

/// Register every scalar function on the worker.
///
/// The worker's own version is *not* exposed as a scalar function: it is carried
/// as the catalog's `implementation_version` (see `main.rs`), which an agent
/// reads from `vgi_catalogs()` without spending a query and which can never drift
/// from the running build.
pub fn register(worker: &mut Worker) {
    worker.register_scalar(encoding::EncodingForModel);

    // count_tokens(text) and count_tokens(text, model).
    worker.register_scalar(count::CountTokens);
    worker.register_scalar(count::CountTokensModel);

    // tokenize(text) and tokenize(text, model) -> INTEGER[].
    worker.register_scalar(tokenize::Tokenize);
    worker.register_scalar(tokenize::TokenizeModel);

    // truncate_to_tokens(text, n) and (text, n, model) -> VARCHAR.
    worker.register_scalar(truncate::TruncateToTokens);
    worker.register_scalar(truncate::TruncateToTokensModel);

    // chunk_by_tokens(text, max_tokens) and (text, max_tokens, overlap) -> VARCHAR[].
    worker.register_scalar(chunk::ChunkByTokens);
    worker.register_scalar(chunk::ChunkByTokensOverlap);
}
