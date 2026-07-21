//! `chunk_by_tokens(text, max_tokens) -> VARCHAR[]` and
//! `chunk_by_tokens(text, max_tokens, overlap) -> VARCHAR[]` — split `text` into
//! chunks of at most `max_tokens` tokens (under cl100k_base), optionally sharing
//! `overlap` tokens between consecutive chunks (RAG windows). Each chunk decodes
//! to a valid string.
//!
//! NULL text → NULL list. Empty text or `max_tokens <= 0` → empty list. `overlap`
//! is clamped to `max_tokens - 1` so the window always advances.

use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::{ArrayRef, RecordBatch};
use vgi::{
    ArgSpec, BindParams, BindResponse, FunctionExample, FunctionMetadata, ProcessParams,
    ScalarFunction,
};
use vgi_rpc::{Result, RpcError};

use crate::arrow_io::{
    finish_list_varchar, int_val, list_varchar_builder, list_varchar_type, text_str,
};
use crate::tiktoken::{self, Encoding};

/// Guaranteed-runnable, catalog-qualified examples (VGI509). Each `sql` is
/// self-contained and re-runnable against an attached `tiktoken` worker. These
/// span the worker's surface (count/tokenize/encoding/truncate/chunk) and all
/// use only the bundled BPE encodings, so they execute with no network access.
/// We omit `expected_result` deliberately — the linter only needs each query to
/// execute cleanly.
const EXECUTABLE_EXAMPLES: &str = r#"[
  {
    "description": "Count the GPT-4/3.5 (cl100k_base) tokens in a sentence.",
    "sql": "SELECT tiktoken.main.count_tokens('The quick brown fox jumps over the lazy dog.') AS n"
  },
  {
    "description": "Count tokens with a specific model's encoding (gpt-4o uses o200k_base).",
    "sql": "SELECT tiktoken.main.count_tokens('Summarize this prompt.', 'gpt-4o') AS n"
  },
  {
    "description": "Get the BPE token ids for a string under the default encoding.",
    "sql": "SELECT tiktoken.main.tokenize('tiktoken is great!') AS ids"
  },
  {
    "description": "Map a model name to its tiktoken encoding name.",
    "sql": "SELECT tiktoken.main.encoding_for_model('gpt-4o') AS encoding"
  },
  {
    "description": "Truncate text to its first 5 tokens, decoded back to a string.",
    "sql": "SELECT tiktoken.main.truncate_to_tokens('The quick brown fox jumps over the lazy dog.', 5) AS clipped"
  },
  {
    "description": "Split a document into non-overlapping 8-token chunks for embedding.",
    "sql": "SELECT tiktoken.main.chunk_by_tokens('A long document to split before embedding for retrieval augmented generation.', 8) AS chunks"
  },
  {
    "description": "Split a document into 8-token windows that share 2 overlapping tokens (RAG).",
    "sql": "SELECT tiktoken.main.chunk_by_tokens('A long document to split into overlapping windows for retrieval.', 8, 2) AS chunks"
  }
]"#;

/// The `vgi.example_queries` payload shared by both `chunk_by_tokens` arity
/// overloads (see the note on `count_examples_json`): overloads collapse to one
/// `duckdb_functions()` row whose native examples are the union of both, so the
/// tag must describe every native SQL to satisfy VGI515.
fn chunk_examples_json() -> String {
    crate::meta::example_queries_json(&[
        (
            "Split a document into non-overlapping windows of at most 256 tokens (cl100k_base) for embedding.",
            "SELECT tiktoken.main.chunk_by_tokens('A long document to split before embedding.', 256);",
        ),
        (
            "Split a document into 256-token windows that share 32 overlapping tokens, preserving context across chunks for RAG.",
            "SELECT tiktoken.main.chunk_by_tokens('A long RAG document to split into overlapping windows.', 256, 32);",
        ),
    ])
}

/// Shared chunk builder over an already-validated `(text, max, overlap)`.
fn append_chunks(builder: &mut ListBuilder<StringBuilder>, text: &str, max: usize, overlap: usize) {
    for chunk in tiktoken::chunk(text, max, overlap, Encoding::default_encoding()) {
        builder.values().append_value(&chunk);
    }
    builder.append(true);
}

/// `chunk_by_tokens(text, max_tokens)` — no overlap.
pub struct ChunkByTokens;

impl ScalarFunction for ChunkByTokens {
    fn name(&self) -> &str {
        "chunk_by_tokens"
    }

    fn metadata(&self) -> FunctionMetadata {
        FunctionMetadata {
            description: "Split text into chunks of at most max_tokens tokens (cl100k_base), with \
                          no overlap, as VARCHAR[]. Each chunk decodes to valid text. NULL -> NULL"
                .into(),
            return_type: Some(list_varchar_type()),
            examples: vec![FunctionExample {
                sql: "SELECT tiktoken.main.chunk_by_tokens('A long document to split before embedding.', 256);".into(),
                description: "Split a document into non-overlapping windows of at most 256 tokens (cl100k_base) for embedding.".into(),
                expected_output: None,
            }],
            tags: {
                let mut tags = crate::meta::object_tags(
                    "Chunk By Tokens (No Overlap)",
                    "Split text into chunks of at most max_tokens tokens (under cl100k_base) with \
                     no overlap, returned as a `VARCHAR[]`. Each chunk decodes to valid text. NULL \
                     text -> NULL; empty text or max_tokens <= 0 -> []. Use to window long \
                     documents before embedding for RAG.",
                    "Split text into non-overlapping token-bounded chunks as `VARCHAR[]`. \
                     `chunk_by_tokens(text, 256)`.",
                    &[
                        "chunk",
                        "chunk by tokens",
                        "split text",
                        "token windows",
                        "rag chunking",
                        "document chunks",
                        "embedding chunks",
                        "cl100k_base",
                    ],
                    "shape",
                );
                tags.push(("vgi.example_queries".into(), chunk_examples_json()));
                tags.push(("vgi.executable_examples".into(), EXECUTABLE_EXAMPLES.into()));
                tags
            },
            ..Default::default()
        }
    }

    fn argument_specs(&self) -> Vec<ArgSpec> {
        vec![
            ArgSpec::column("text", 0, "varchar", "The text to split into chunks."),
            ArgSpec::column(
                "max_tokens",
                1,
                "int32",
                "The maximum number of tokens allowed in each chunk. A value of 0 or less yields \
                 an empty list.",
            ),
        ]
    }

    fn on_bind(&self, _params: &BindParams) -> Result<BindResponse> {
        Ok(BindResponse::result(list_varchar_type()))
    }

    fn process(&self, params: &ProcessParams, batch: &RecordBatch) -> Result<RecordBatch> {
        let text = batch.column(0);
        let max = batch.column(1);
        let rows = batch.num_rows();
        let mut builder = list_varchar_builder();
        for i in 0..rows {
            match (text_str(text, i)?, int_val(max, i)?) {
                (Some(t), Some(m)) => append_chunks(&mut builder, t, m.max(0) as usize, 0),
                _ => builder.append_null(),
            }
        }
        let arr: ArrayRef = finish_list_varchar(builder);
        debug_assert_eq!(arr.data_type(), &list_varchar_type());
        RecordBatch::try_new(params.output_schema.clone(), vec![arr])
            .map_err(|e| RpcError::runtime_error(e.to_string()))
    }
}

/// `chunk_by_tokens(text, max_tokens, overlap)` — with token overlap.
pub struct ChunkByTokensOverlap;

impl ScalarFunction for ChunkByTokensOverlap {
    fn name(&self) -> &str {
        "chunk_by_tokens"
    }

    fn metadata(&self) -> FunctionMetadata {
        FunctionMetadata {
            description: "Split text into chunks of at most max_tokens tokens (cl100k_base) with \
                          `overlap` tokens shared between consecutive chunks (RAG windows), as \
                          VARCHAR[]. overlap is clamped to max_tokens-1. NULL -> NULL"
                .into(),
            return_type: Some(list_varchar_type()),
            examples: vec![FunctionExample {
                sql: "SELECT tiktoken.main.chunk_by_tokens('A long RAG document to split into overlapping windows.', 256, 32);".into(),
                description: "Split a document into 256-token windows that share 32 overlapping tokens, preserving context across chunks for RAG.".into(),
                expected_output: None,
            }],
            tags: {
                let mut tags = crate::meta::object_tags(
                "Chunk By Tokens With Overlap",
                "Split text into chunks of at most max_tokens tokens (under cl100k_base) with \
                 `overlap` tokens shared between consecutive chunks (sliding RAG windows), \
                 returned as a `VARCHAR[]`. overlap is clamped to max_tokens-1 so the window always \
                 advances. NULL -> NULL. Use to preserve context across chunk boundaries.",
                "Split text into overlapping token-bounded chunks as `VARCHAR[]`. \
                 `chunk_by_tokens(text, 256, 32)`.",
                &[
                    "chunk",
                    "chunk by tokens",
                    "overlapping chunks",
                    "sliding window",
                    "rag chunking",
                    "token windows",
                    "document chunks",
                    "overlap",
                    "cl100k_base",
                ],
                "shape",
                );
                tags.push(("vgi.example_queries".into(), chunk_examples_json()));
                tags
            },
            ..Default::default()
        }
    }

    fn argument_specs(&self) -> Vec<ArgSpec> {
        vec![
            ArgSpec::column("text", 0, "varchar", "The text to split into chunks."),
            ArgSpec::column(
                "max_tokens",
                1,
                "int32",
                "The maximum number of tokens allowed in each chunk. A value of 0 or less yields \
                 an empty list.",
            ),
            ArgSpec::column(
                "overlap",
                2,
                "int32",
                "The number of tokens each chunk shares with the previous one (sliding RAG \
                 windows); it is clamped to max_tokens minus 1 so the window always advances.",
            ),
        ]
    }

    fn on_bind(&self, _params: &BindParams) -> Result<BindResponse> {
        Ok(BindResponse::result(list_varchar_type()))
    }

    fn process(&self, params: &ProcessParams, batch: &RecordBatch) -> Result<RecordBatch> {
        let text = batch.column(0);
        let max = batch.column(1);
        let overlap = batch.column(2);
        let rows = batch.num_rows();
        let mut builder = list_varchar_builder();
        for i in 0..rows {
            match (text_str(text, i)?, int_val(max, i)?, int_val(overlap, i)?) {
                (Some(t), Some(m), Some(o)) => {
                    append_chunks(&mut builder, t, m.max(0) as usize, o.max(0) as usize)
                }
                _ => builder.append_null(),
            }
        }
        let arr: ArrayRef = finish_list_varchar(builder);
        debug_assert_eq!(arr.data_type(), &list_varchar_type());
        RecordBatch::try_new(params.output_schema.clone(), vec![arr])
            .map_err(|e| RpcError::runtime_error(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow_io::test_support::{process_params, varchar_list_row};
    use arrow_array::{Array, Int32Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    const DOC: &str = "The quick brown fox jumps over the lazy dog. \
                       Pack my box with five dozen liquor jugs. \
                       How vexingly quick daft zebras jump!";

    fn run_overlap(text: &str, max: i32, overlap: i32) -> ArrayRef {
        let t: ArrayRef = Arc::new(StringArray::from(vec![Some(text)]));
        let m: ArrayRef = Arc::new(Int32Array::from(vec![Some(max)]));
        let o: ArrayRef = Arc::new(Int32Array::from(vec![Some(overlap)]));
        let schema = Arc::new(Schema::new(vec![
            Field::new("text", DataType::Utf8, true),
            Field::new("max", DataType::Int32, true),
            Field::new("overlap", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![t, m, o]).unwrap();
        let bind = BindParams {
            input_schema: Some(schema),
            ..Default::default()
        };
        let bound = ChunkByTokensOverlap.on_bind(&bind).unwrap();
        let params = process_params(bound.output_schema, vgi::arguments::Arguments::default());
        ChunkByTokensOverlap
            .process(&params, &batch)
            .unwrap()
            .column(0)
            .clone()
    }

    #[test]
    fn chunks_each_under_max_and_overlap_covers() {
        let enc = Encoding::Cl100kBase;
        let out = run_overlap(DOC, 8, 2);
        let chunks = varchar_list_row(&out, 0);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(tiktoken::count(c, enc) <= 8, "chunk over max: {c:?}");
        }
        // Reconstruct the token stream from chunks (minus overlap) == original.
        let full = tiktoken::tokenize(DOC, enc);
        let mut rebuilt: Vec<u32> = Vec::new();
        for (i, c) in chunks.iter().enumerate() {
            let ct = tiktoken::tokenize(c, enc);
            if i == 0 {
                rebuilt.extend_from_slice(&ct);
            } else {
                rebuilt.extend_from_slice(&ct[2.min(ct.len())..]);
            }
        }
        assert_eq!(rebuilt, full, "overlapped chunks must cover the input");
    }

    #[test]
    fn no_overlap_two_arity() {
        let t: ArrayRef = Arc::new(StringArray::from(vec![Some(DOC), None]));
        let m: ArrayRef = Arc::new(Int32Array::from(vec![Some(8), Some(8)]));
        let schema = Arc::new(Schema::new(vec![
            Field::new("text", DataType::Utf8, true),
            Field::new("max", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![t, m]).unwrap();
        let bind = BindParams {
            input_schema: Some(schema),
            ..Default::default()
        };
        let bound = ChunkByTokens.on_bind(&bind).unwrap();
        let params = process_params(bound.output_schema, vgi::arguments::Arguments::default());
        let out = ChunkByTokens
            .process(&params, &batch)
            .unwrap()
            .column(0)
            .clone();
        let chunks = varchar_list_row(&out, 0);
        // Concatenating no-overlap chunks reconstructs the whole text.
        assert_eq!(chunks.concat(), DOC);
        assert!(out.is_null(1), "NULL text -> NULL list");
    }
}
