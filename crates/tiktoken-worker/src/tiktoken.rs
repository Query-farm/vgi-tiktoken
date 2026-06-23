//! Pure tokenization engine (no Arrow, no RPC).
//!
//! Wraps [`tiktoken-rs`](https://crates.io/crates/tiktoken-rs), which **bundles**
//! the OpenAI byte-pair encodings (cl100k_base, o200k_base, p50k_base,
//! r50k_base, …) directly in the binary — there is **no network download** at
//! runtime. Each encoder is built lazily on first use and cached for the
//! process lifetime via [`once_cell`].
//!
//! ## Encodings & model mapping
//!
//! A *model name* (e.g. `"gpt-4o"`) maps to an *encoding* (e.g. `"o200k_base"`)
//! via [`encoding_for_model`]. The default encoding when no model is supplied is
//! **cl100k_base** (GPT-4 / GPT-3.5-turbo family). An **unknown model** is a
//! documented, recoverable condition — the resolver returns `None`, and the
//! Arrow adapters translate that to a SQL **NULL** (never an error), so dirty
//! data does not abort a scan.
//!
//! ## Accuracy
//!
//! Counts are **exact** for OpenAI BPE families (GPT-4/4o/4.1/5, o-series,
//! GPT-3.5, the legacy davinci/codex models). For **other** model families
//! (Anthropic Claude, Meta Llama, Mistral, Google Gemini, …) tiktoken is a
//! *close proxy* but not exact — those vendors use different tokenizers, so
//! treat the result as an estimate.

use once_cell::sync::OnceCell;
use tiktoken_rs::tokenizer::Tokenizer;
use tiktoken_rs::{
    bpe_for_tokenizer, cl100k_base_singleton, o200k_base_singleton, p50k_base_singleton,
    r50k_base_singleton, CoreBPE,
};

/// The default encoding used when a caller supplies no model: cl100k_base, the
/// tokenizer of the GPT-4 / GPT-3.5-turbo family.
pub const DEFAULT_ENCODING: &str = "cl100k_base";

/// A canonical, supported encoding name. We expose the four bundled BPE
/// encodings most callers want plus the harmony variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    Cl100kBase,
    O200kBase,
    P50kBase,
    R50kBase,
    O200kHarmony,
}

impl Encoding {
    /// The default encoding used when no model/encoding is supplied: the one
    /// named by [`DEFAULT_ENCODING`] (cl100k_base).
    pub fn default_encoding() -> Encoding {
        Encoding::from_encoding_name(DEFAULT_ENCODING)
            .expect("DEFAULT_ENCODING is a valid encoding name")
    }

    /// The canonical encoding name as a string (e.g. `"cl100k_base"`).
    pub fn name(self) -> &'static str {
        match self {
            Encoding::Cl100kBase => "cl100k_base",
            Encoding::O200kBase => "o200k_base",
            Encoding::P50kBase => "p50k_base",
            Encoding::R50kBase => "r50k_base",
            Encoding::O200kHarmony => "o200k_harmony",
        }
    }

    /// Resolve a canonical encoding name directly to an [`Encoding`]. Accepts the
    /// names this worker emits; matching is case-insensitive.
    pub fn from_encoding_name(name: &str) -> Option<Encoding> {
        match name.trim().to_ascii_lowercase().as_str() {
            "cl100k_base" | "cl100k" => Some(Encoding::Cl100kBase),
            "o200k_base" | "o200k" => Some(Encoding::O200kBase),
            "p50k_base" | "p50k" => Some(Encoding::P50kBase),
            "r50k_base" | "r50k" | "gpt2" => Some(Encoding::R50kBase),
            "o200k_harmony" => Some(Encoding::O200kHarmony),
            _ => None,
        }
    }

    fn tokenizer(self) -> Tokenizer {
        match self {
            Encoding::Cl100kBase => Tokenizer::Cl100kBase,
            Encoding::O200kBase => Tokenizer::O200kBase,
            Encoding::P50kBase => Tokenizer::P50kBase,
            Encoding::R50kBase => Tokenizer::R50kBase,
            Encoding::O200kHarmony => Tokenizer::O200kHarmony,
        }
    }

    /// Borrow the lazily-built, process-cached [`CoreBPE`] for this encoding.
    fn bpe(self) -> &'static CoreBPE {
        // Each branch caches its own singleton. The tiktoken-rs `*_singleton`
        // helpers already memoize, but we additionally cache the *resolved*
        // reference so we never re-hash the encoding name on the hot path.
        match self {
            Encoding::Cl100kBase => {
                static C: OnceCell<&'static CoreBPE> = OnceCell::new();
                C.get_or_init(cl100k_base_singleton)
            }
            Encoding::O200kBase => {
                static C: OnceCell<&'static CoreBPE> = OnceCell::new();
                C.get_or_init(o200k_base_singleton)
            }
            Encoding::P50kBase => {
                static C: OnceCell<&'static CoreBPE> = OnceCell::new();
                C.get_or_init(p50k_base_singleton)
            }
            Encoding::R50kBase => {
                static C: OnceCell<&'static CoreBPE> = OnceCell::new();
                C.get_or_init(r50k_base_singleton)
            }
            Encoding::O200kHarmony => {
                static C: OnceCell<&'static CoreBPE> = OnceCell::new();
                // Built once from its Tokenizer (still bundled, no download).
                C.get_or_init(|| {
                    bpe_for_tokenizer(Encoding::O200kHarmony.tokenizer())
                        .expect("o200k_harmony is bundled in tiktoken-rs")
                })
            }
        }
    }
}

/// Resolve a *model name* to the [`Encoding`] it uses, or `None` for a model we
/// do not recognize. Delegates to tiktoken-rs's model→tokenizer table (which
/// covers the OpenAI families incl. gpt-4o/4.1/5, the o-series, gpt-3.5-turbo,
/// and the legacy davinci/codex models), with the bare-family names mapped
/// explicitly so `"gpt-4o"`, `"gpt-4"`, `"gpt-3.5-turbo"` resolve cleanly.
pub fn encoding_for_model(model: &str) -> Option<Encoding> {
    let m = model.trim();
    if m.is_empty() {
        return None;
    }
    // Fast path for the common bare names callers pass.
    match m.to_ascii_lowercase().as_str() {
        "gpt-4o" | "gpt-4o-mini" | "gpt-4.1" | "gpt-4.5" => return Some(Encoding::O200kBase),
        "gpt-4" | "gpt-4-turbo" | "gpt-3.5-turbo" | "gpt-3.5" | "chatgpt" => {
            return Some(Encoding::Cl100kBase)
        }
        _ => {}
    }
    let tok = tiktoken_rs::tokenizer::get_tokenizer(m)?;
    Some(match tok {
        Tokenizer::Cl100kBase => Encoding::Cl100kBase,
        Tokenizer::O200kBase => Encoding::O200kBase,
        Tokenizer::P50kBase | Tokenizer::P50kEdit => Encoding::P50kBase,
        // r50k_base is the GPT-2 encoding; tiktoken-rs models it as both
        // `R50kBase` and `Gpt2`.
        Tokenizer::R50kBase | Tokenizer::Gpt2 => Encoding::R50kBase,
        Tokenizer::O200kHarmony => Encoding::O200kHarmony,
    })
}

/// Resolve a *model name* to its encoding **name** (e.g. `"gpt-4o"` →
/// `"o200k_base"`), or `None` for an unknown model.
pub fn encoding_name_for_model(model: &str) -> Option<&'static str> {
    encoding_for_model(model).map(Encoding::name)
}

/// Resolve a user-supplied identifier — either a model name *or* a canonical
/// encoding name — to an [`Encoding`]. We try the encoding name first (so
/// callers can pass `"o200k_base"` directly), then fall back to the model table.
/// Returns `None` for anything unrecognized.
pub fn resolve(identifier: &str) -> Option<Encoding> {
    Encoding::from_encoding_name(identifier).or_else(|| encoding_for_model(identifier))
}

/// Count BPE tokens in `text` under `encoding`. Empty text → 0.
///
/// Uses `encode_ordinary` (special tokens are treated as ordinary text), which
/// is the correct, deterministic count for arbitrary input.
pub fn count(text: &str, encoding: Encoding) -> usize {
    if text.is_empty() {
        return 0;
    }
    encoding.bpe().encode_ordinary(text).len()
}

/// Tokenize `text` to its BPE token ids under `encoding`. Empty text → empty.
pub fn tokenize(text: &str, encoding: Encoding) -> Vec<u32> {
    if text.is_empty() {
        return Vec::new();
    }
    encoding.bpe().encode_ordinary(text)
}

/// Decode token ids back to text under `encoding`. Invalid UTF-8 produced by a
/// mid-character token boundary is replaced lossily (so the result is always a
/// valid `String`).
pub fn decode(tokens: &[u32], encoding: Encoding) -> String {
    if tokens.is_empty() {
        return String::new();
    }
    let bytes = encoding
        .bpe()
        .decode_bytes(tokens)
        .unwrap_or_else(|_| Vec::new());
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Keep only the first `n` tokens of `text` and decode them back to a string.
/// `n == 0` → empty string; `n >= count` → the whole text round-trips.
pub fn truncate(text: &str, n: usize, encoding: Encoding) -> String {
    if n == 0 || text.is_empty() {
        return String::new();
    }
    let tokens = encoding.bpe().encode_ordinary(text);
    let end = n.min(tokens.len());
    decode(&tokens[..end], encoding)
}

/// Split `text` into chunks of at most `max_tokens` tokens, with `overlap`
/// tokens shared between consecutive chunks (for RAG windows). Each chunk is
/// decoded back to a valid string. Concatenating the chunks (minus the overlap)
/// covers the whole input.
///
/// - `max_tokens == 0` is invalid → returns an empty `Vec` (the adapter maps
///   this to an empty list; chunking into zero-size windows is meaningless).
/// - `overlap` is clamped to `max_tokens - 1` so the window always advances and
///   the function terminates.
/// - Empty text → empty `Vec`.
pub fn chunk(text: &str, max_tokens: usize, overlap: usize, encoding: Encoding) -> Vec<String> {
    if text.is_empty() || max_tokens == 0 {
        return Vec::new();
    }
    let tokens = encoding.bpe().encode_ordinary(text);
    if tokens.is_empty() {
        return Vec::new();
    }
    // Window must advance by at least one token.
    let overlap = overlap.min(max_tokens.saturating_sub(1));
    let step = max_tokens - overlap;

    let mut out = Vec::new();
    let mut start = 0usize;
    while start < tokens.len() {
        let end = (start + max_tokens).min(tokens.len());
        out.push(decode(&tokens[start..end], encoding));
        if end == tokens.len() {
            break;
        }
        start += step;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_cl100k() {
        assert_eq!(DEFAULT_ENCODING, "cl100k_base");
        assert_eq!(Encoding::Cl100kBase.name(), "cl100k_base");
    }

    #[test]
    fn model_to_encoding() {
        assert_eq!(encoding_name_for_model("gpt-4o"), Some("o200k_base"));
        assert_eq!(encoding_name_for_model("gpt-4"), Some("cl100k_base"));
        assert_eq!(
            encoding_name_for_model("gpt-3.5-turbo"),
            Some("cl100k_base")
        );
        assert_eq!(
            encoding_name_for_model("text-davinci-003"),
            Some("p50k_base")
        );
        assert_eq!(encoding_name_for_model("definitely-not-a-model"), None);
        assert_eq!(encoding_name_for_model(""), None);
    }

    #[test]
    fn resolve_accepts_encoding_or_model() {
        assert_eq!(resolve("o200k_base"), Some(Encoding::O200kBase));
        assert_eq!(resolve("gpt-4o"), Some(Encoding::O200kBase));
        assert_eq!(resolve("cl100k_base"), Some(Encoding::Cl100kBase));
        assert_eq!(resolve("nonsense"), None);
    }

    #[test]
    fn hello_world_is_two_tokens_cl100k() {
        // Known, deterministic tiktoken count.
        assert_eq!(count("hello world", Encoding::Cl100kBase), 2);
    }

    #[test]
    fn empty_text_is_zero_and_empty() {
        assert_eq!(count("", Encoding::Cl100kBase), 0);
        assert!(tokenize("", Encoding::Cl100kBase).is_empty());
        assert!(chunk("", 10, 0, Encoding::Cl100kBase).is_empty());
        assert_eq!(truncate("", 5, Encoding::Cl100kBase), "");
    }

    #[test]
    fn tokenize_known_sequence() {
        // "tiktoken is great!" under cl100k_base — count is stable and the
        // tokens round-trip through decode.
        let toks = tokenize("tiktoken is great!", Encoding::Cl100kBase);
        assert_eq!(
            toks.len(),
            count("tiktoken is great!", Encoding::Cl100kBase)
        );
        assert!(toks.len() >= 4);
        assert_eq!(decode(&toks, Encoding::Cl100kBase), "tiktoken is great!");
    }

    #[test]
    fn truncate_is_a_token_prefix() {
        let text = "The quick brown fox jumps over the lazy dog.";
        let full = count(text, Encoding::Cl100kBase);
        // Truncating to >= full round-trips the whole text.
        assert_eq!(truncate(text, full + 100, Encoding::Cl100kBase), text);
        // Truncating to 3 tokens yields exactly the first 3 tokens decoded.
        let three = truncate(text, 3, Encoding::Cl100kBase);
        assert_eq!(count(&three, Encoding::Cl100kBase), 3);
        assert!(text.starts_with(&three));
        // n == 0 → empty.
        assert_eq!(truncate(text, 0, Encoding::Cl100kBase), "");
    }

    #[test]
    fn chunk_respects_max_and_overlap_and_covers() {
        let text = "The quick brown fox jumps over the lazy dog. \
                    Pack my box with five dozen liquor jugs. \
                    How vexingly quick daft zebras jump!";
        let enc = Encoding::Cl100kBase;
        let total = count(text, enc);
        assert!(total > 10, "need a non-trivial token count for the test");

        let max = 8;
        let overlap = 2;
        let chunks = chunk(text, max, overlap, enc);
        assert!(chunks.len() > 1, "must split into multiple chunks");

        // Every chunk is <= max tokens.
        for c in &chunks {
            assert!(count(c, enc) <= max, "chunk over max: {c:?}");
        }

        // Re-tokenizing all chunks and stitching with the declared overlap must
        // reconstruct exactly the original token stream (coverage + correct
        // overlap).
        let full = tokenize(text, enc);
        let mut rebuilt: Vec<u32> = Vec::new();
        for (i, c) in chunks.iter().enumerate() {
            let ct = tokenize(c, enc);
            if i == 0 {
                rebuilt.extend_from_slice(&ct);
            } else {
                // Drop the leading `overlap` tokens that repeat the prior tail.
                rebuilt.extend_from_slice(&ct[overlap.min(ct.len())..]);
            }
        }
        assert_eq!(rebuilt, full, "chunks (minus overlap) must cover the input");
    }

    #[test]
    fn chunk_overlap_clamped_and_zero_max_empty() {
        let enc = Encoding::Cl100kBase;
        // overlap >= max is clamped so the window still advances (terminates).
        let chunks = chunk("one two three four five six seven eight", 3, 99, enc);
        assert!(!chunks.is_empty());
        for c in &chunks {
            assert!(count(c, enc) <= 3);
        }
        // max_tokens == 0 → empty.
        assert!(chunk("anything", 0, 0, enc).is_empty());
    }

    #[test]
    fn counts_differ_across_encodings_but_are_deterministic() {
        let text = "Tokenization across encodings.";
        let a = count(text, Encoding::Cl100kBase);
        let b = count(text, Encoding::O200kBase);
        // Both are stable, positive counts.
        assert!(a > 0 && b > 0);
        assert_eq!(a, count(text, Encoding::Cl100kBase));
    }
}
