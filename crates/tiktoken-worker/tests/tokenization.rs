//! Integration tests for the pure tokenization engine against KNOWN,
//! deterministic tiktoken results. These exercise `tiktoken-worker`'s public
//! engine the same way the Arrow adapters do, but without any Arrow/RPC plumbing.

use tiktoken_worker::tiktoken::{self, Encoding};

#[test]
fn hello_world_is_two_tokens() {
    // The canonical known count: cl100k_base "hello world" = 2 tokens.
    assert_eq!(tiktoken::count("hello world", Encoding::Cl100kBase), 2);
}

#[test]
fn empty_text_is_zero_and_empty() {
    assert_eq!(tiktoken::count("", Encoding::Cl100kBase), 0);
    assert!(tiktoken::tokenize("", Encoding::Cl100kBase).is_empty());
    assert!(tiktoken::chunk("", 16, 4, Encoding::Cl100kBase).is_empty());
    assert_eq!(tiktoken::truncate("", 4, Encoding::Cl100kBase), "");
}

#[test]
fn model_to_encoding_mapping() {
    assert_eq!(
        tiktoken::encoding_name_for_model("gpt-4o"),
        Some("o200k_base")
    );
    assert_eq!(
        tiktoken::encoding_name_for_model("gpt-4"),
        Some("cl100k_base")
    );
    assert_eq!(
        tiktoken::encoding_name_for_model("gpt-3.5-turbo"),
        Some("cl100k_base")
    );
    assert_eq!(tiktoken::encoding_name_for_model("totally-made-up"), None);
}

#[test]
fn tokenize_round_trips() {
    let enc = Encoding::Cl100kBase;
    let text = "tiktoken is great!";
    let toks = tiktoken::tokenize(text, enc);
    assert_eq!(toks.len(), tiktoken::count(text, enc));
    assert_eq!(tiktoken::decode(&toks, enc), text);
}

#[test]
fn truncate_is_a_prefix_and_round_trips_whole() {
    let enc = Encoding::Cl100kBase;
    let text = "The quick brown fox jumps over the lazy dog.";
    let full = tiktoken::count(text, enc);
    assert_eq!(tiktoken::truncate(text, full + 50, enc), text);
    let three = tiktoken::truncate(text, 3, enc);
    assert_eq!(tiktoken::count(&three, enc), 3);
    assert!(text.starts_with(&three));
}

#[test]
fn chunk_covers_with_correct_overlap_and_under_max() {
    let enc = Encoding::Cl100kBase;
    let text = "The quick brown fox jumps over the lazy dog. \
                Pack my box with five dozen liquor jugs. \
                How vexingly quick daft zebras jump!";
    let max = 8;
    let overlap = 2;
    let chunks = tiktoken::chunk(text, max, overlap, enc);
    assert!(chunks.len() > 1);
    for c in &chunks {
        assert!(tiktoken::count(c, enc) <= max);
    }
    // Stitch chunks (dropping the repeated overlap) → original token stream.
    let full = tiktoken::tokenize(text, enc);
    let mut rebuilt: Vec<u32> = Vec::new();
    for (i, c) in chunks.iter().enumerate() {
        let ct = tiktoken::tokenize(c, enc);
        if i == 0 {
            rebuilt.extend_from_slice(&ct);
        } else {
            rebuilt.extend_from_slice(&ct[overlap.min(ct.len())..]);
        }
    }
    assert_eq!(rebuilt, full);
}
