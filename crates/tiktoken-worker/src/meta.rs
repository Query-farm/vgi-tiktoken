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

/// Render `(description, sql)` pairs as the JSON list of `{"description", "sql"}`
/// objects that `vgi.example_queries` expects (VGI515 requires every example to
/// carry a non-empty description; the native `duckdb_functions().examples`
/// column can only carry SQL, so the human-readable description must ride on
/// this tag). Each field is JSON-escaped, mirroring [`keywords_json`].
pub fn example_queries_json(examples: &[(&str, &str)]) -> String {
    fn push_escaped(out: &mut String, s: &str) {
        out.push('"');
        for ch in s.chars() {
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

    let mut out = String::from("[");
    for (i, (description, sql)) in examples.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("{\"description\":");
        push_escaped(&mut out, description);
        out.push_str(",\"sql\":");
        push_escaped(&mut out, sql);
        out.push('}');
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

// Composed examples: realistic queries that combine two or more of the worker's
// functions. Each is defined once here and referenced from the schema-level
// `vgi.example_queries` list (lib.rs) and from the example list of the function
// that drives the query, so the copies can never drift apart.

/// `chunk_by_tokens` feeding `count_tokens`: window a document, then verify each
/// chunk's token count (driven by `chunk_by_tokens`).
pub const COMPOSED_CHUNK_TOKEN_COUNTS: (&str, &str) = (
    "Split a document into 8-token chunks (cl100k_base), number them in order with WITH ORDINALITY, and count the tokens in each chunk to confirm it fits the budget (4 chunks of 8, 8, 8 and 5 tokens).",
    "SELECT c.chunk_no, c.chunk, tiktoken.main.count_tokens(c.chunk) AS tokens FROM unnest(tiktoken.main.chunk_by_tokens('Retrieval augmented generation splits long documents into token-bounded chunks, embeds each chunk, and retrieves the most relevant ones at query time.', 8)) WITH ORDINALITY AS c(chunk, chunk_no) ORDER BY c.chunk_no;",
);

/// `encoding_for_model` beside `count_tokens(text, model)`: compare one text's
/// token cost across models (driven by `count_tokens`).
pub const COMPOSED_MODEL_COST_COMPARISON: (&str, &str) = (
    "Compare what one multilingual sentence costs under three models, showing each model's encoding: gpt-4o (o200k_base) needs 20 tokens versus 24 for gpt-4 (cl100k_base) and 30 for text-davinci-003 (p50k_base).",
    "SELECT m.model, tiktoken.main.encoding_for_model(m.model) AS encoding, tiktoken.main.count_tokens('Tokenizers differ: GPT-4o packs multilingual text such as こんにちは世界 into fewer tokens.', m.model) AS tokens FROM (VALUES ('gpt-4o'), ('gpt-4'), ('text-davinci-003')) AS m(model) ORDER BY tokens, m.model;",
);

/// `truncate_to_tokens` checked with `count_tokens`: fit prompts to a budget and
/// report before/after counts (driven by `truncate_to_tokens`).
pub const COMPOSED_PROMPT_BUDGET: (&str, &str) = (
    "Fit a table of prompts to a 10-token budget: report each prompt's original token count, the clipped text, and its token count after truncation (the 3-token prompt passes through unchanged; the 20-token one is cut to 10).",
    "SELECT p.id, tiktoken.main.count_tokens(p.prompt) AS original_tokens, tiktoken.main.truncate_to_tokens(p.prompt, 10) AS clipped, tiktoken.main.count_tokens(tiktoken.main.truncate_to_tokens(p.prompt, 10)) AS clipped_tokens FROM (VALUES (1, 'Short question?'), (2, 'Please summarize the following quarterly report in three bullet points, focusing on revenue growth, margins and guidance.')) AS p(id, prompt) ORDER BY p.id;",
);
