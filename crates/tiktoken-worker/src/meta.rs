//! Shared helpers for the per-object discovery/description metadata that the
//! `vgi-lint` strict profile expects on **every** function.
//!
//! Each function surfaces these in its `FunctionMetadata.tags`:
//! - `vgi.title` (VGI124)        — human-friendly display name
//! - `vgi.doc_llm` (VGI112) — Markdown narrative aimed at LLMs/agents
//! - `vgi.doc_md` (VGI113)  — Markdown narrative for human docs
//! - `vgi.keywords` (VGI126)        — search terms/synonyms as a JSON array of strings
//!
//! Per-object `vgi.source_url` is intentionally **not** emitted here: VGI139
//! requires `source_url` to live only on the catalog object, so provenance is
//! recorded once at the catalog level (see `main.rs`).

/// Render a list of keyword strings as a JSON array literal (VGI138 expects
/// `vgi.keywords` to be a JSON array of strings, not a comma-separated string).
///
/// Each term is JSON-escaped so quotes/backslashes/control characters in a
/// keyword can never produce malformed JSON.
pub fn keywords_json(keywords: &[&str]) -> String {
    let mut out = String::from("[");
    for (i, kw) in keywords.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('"');
        for ch in kw.chars() {
            match ch {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
        out.push('"');
    }
    out.push(']');
    out
}

/// Build the standard per-object discovery/description tags.
///
/// `keywords` is the list of search terms/synonyms for this object; it is
/// serialized to a JSON array of strings for `vgi.keywords` (VGI138).
/// `category` names one of the schema's `vgi.categories` (VGI413) — it groups
/// the object under a navigation section.
pub fn object_tags(
    title: &str,
    description_llm: &str,
    description_md: &str,
    keywords: &[&str],
    category: &str,
) -> Vec<(String, String)> {
    vec![
        ("vgi.title".to_string(), title.to_string()),
        ("vgi.doc_llm".to_string(), description_llm.to_string()),
        ("vgi.doc_md".to_string(), description_md.to_string()),
        ("vgi.keywords".to_string(), keywords_json(keywords)),
        // VGI413: place this object in one of the schema's declared categories.
        ("vgi.category".to_string(), category.to_string()),
    ]
}
