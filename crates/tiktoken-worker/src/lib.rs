//! Library surface of the `tiktoken` VGI worker.
//!
//! The binary (`main.rs`) is the actual worker; this `lib` target exposes the
//! pure tokenization engine so integration tests under `tests/` can exercise it
//! directly, without Arrow or RPC. See [`tiktoken`] for the engine.

pub mod tiktoken;
