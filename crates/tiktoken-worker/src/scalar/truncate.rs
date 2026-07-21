//! `truncate_to_tokens(text, n) -> VARCHAR` (default cl100k_base) and
//! `truncate_to_tokens(text, n, model) -> VARCHAR` — the first `n` tokens of
//! `text`, decoded back to a string.
//!
//! `n <= 0` → empty string. NULL text or NULL `n` → NULL. Unknown model → NULL.

use std::sync::Arc;

use arrow_array::builder::StringBuilder;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::DataType;
use vgi::{
    ArgSpec, BindParams, BindResponse, FunctionExample, FunctionMetadata, ProcessParams,
    ScalarFunction,
};
use vgi_rpc::{Result, RpcError};

use crate::arrow_io::{int_val, text_str};
use crate::tiktoken::{self, Encoding};

/// The `vgi.example_queries` payload shared by both `truncate_to_tokens` arity
/// overloads (see the note on `count_examples_json`): overloads collapse to one
/// `duckdb_functions()` row whose native examples are the union of both, so the
/// tag must describe every native SQL to satisfy VGI515.
fn truncate_examples_json() -> String {
    crate::meta::example_queries_json(&[
        (
            "Keep only the first 5 cl100k_base tokens of the text, decoded back to a string.",
            "SELECT tiktoken.main.truncate_to_tokens('The quick brown fox jumps over the lazy dog.', 5);",
        ),
        (
            "Truncate text to a token budget using a specific model's encoding (gpt-4o uses o200k_base).",
            "SELECT tiktoken.main.truncate_to_tokens('The quick brown fox jumps over the lazy dog.', 5, 'gpt-4o');",
        ),
    ])
}

/// `truncate_to_tokens(text, n)` — first `n` tokens under the default encoding.
pub struct TruncateToTokens;

impl ScalarFunction for TruncateToTokens {
    fn name(&self) -> &str {
        "truncate_to_tokens"
    }

    fn metadata(&self) -> FunctionMetadata {
        FunctionMetadata {
            description: "Keep the first n tokens of text (default cl100k_base) and decode them \
                          back to VARCHAR. n <= 0 -> ''"
                .into(),
            return_type: Some(DataType::Utf8),
            examples: vec![FunctionExample {
                sql: "SELECT tiktoken.main.truncate_to_tokens('The quick brown fox jumps over the lazy dog.', 5);".into(),
                description: "Keep only the first 5 cl100k_base tokens of the text, decoded back to a string.".into(),
                expected_output: None,
            }],
            tags: {
                let mut tags = crate::meta::object_tags(
                "Truncate To Tokens (Default Encoding)",
                "Keep only the first n tokens of text under the default encoding (cl100k_base) \
                 and decode them back to a `VARCHAR`. n <= 0 -> ''; NULL text or NULL n -> NULL. \
                 Use to clip text to a token budget for a prompt or context window.",
                "Truncate text to its first n cl100k_base tokens, decoded back to text. \
                 `truncate_to_tokens(text, 5)`.",
                &[
                    "truncate",
                    "truncate to tokens",
                    "clip tokens",
                    "token budget",
                    "limit tokens",
                    "first n tokens",
                    "context window",
                    "cl100k_base",
                ],
                "shape",
                );
                tags.push(("vgi.example_queries".into(), truncate_examples_json()));
                tags
            },
            ..Default::default()
        }
    }

    fn argument_specs(&self) -> Vec<ArgSpec> {
        vec![
            ArgSpec::column("text", 0, "varchar", "The text to truncate."),
            ArgSpec::column(
                "n",
                1,
                "int32",
                "The maximum number of leading tokens to keep; the text is decoded back from these \
                 tokens. A value of 0 or less yields an empty string.",
            ),
        ]
    }

    fn on_bind(&self, _params: &BindParams) -> Result<BindResponse> {
        Ok(BindResponse::result(DataType::Utf8))
    }

    fn process(&self, params: &ProcessParams, batch: &RecordBatch) -> Result<RecordBatch> {
        let text = batch.column(0);
        let n = batch.column(1);
        let rows = batch.num_rows();
        let mut out = StringBuilder::new();
        for i in 0..rows {
            match (text_str(text, i)?, int_val(n, i)?) {
                (Some(t), Some(n)) => {
                    let n = n.max(0) as usize;
                    out.append_value(tiktoken::truncate(t, n, Encoding::default_encoding()));
                }
                _ => out.append_null(),
            }
        }
        let arr: ArrayRef = Arc::new(out.finish());
        RecordBatch::try_new(params.output_schema.clone(), vec![arr])
            .map_err(|e| RpcError::runtime_error(e.to_string()))
    }
}

/// `truncate_to_tokens(text, n, model)` — first `n` tokens under `model`'s
/// encoding. Unknown model → NULL.
pub struct TruncateToTokensModel;

impl ScalarFunction for TruncateToTokensModel {
    fn name(&self) -> &str {
        "truncate_to_tokens"
    }

    fn metadata(&self) -> FunctionMetadata {
        FunctionMetadata {
            description: "Keep the first n tokens of text using the encoding for the given model \
                          (e.g. 'gpt-4o') and decode them back to VARCHAR. Unknown model -> NULL; \
                          n <= 0 -> ''"
                .into(),
            return_type: Some(DataType::Utf8),
            examples: vec![FunctionExample {
                sql: "SELECT tiktoken.main.truncate_to_tokens('The quick brown fox jumps over the lazy dog.', 5, 'gpt-4o');".into(),
                description: "Truncate text to a token budget using a specific model's encoding (gpt-4o uses o200k_base).".into(),
                expected_output: None,
            }],
            tags: {
                let mut tags = crate::meta::object_tags(
                "Truncate To Tokens For Model",
                "Keep only the first n tokens of text using the encoding for the given model name \
                 (e.g. 'gpt-4o') and decode them back to a `VARCHAR`. Unknown model -> NULL; n <= 0 \
                 -> ''. Use to clip text to a specific model's token budget.",
                "Truncate text to its first n tokens using a model's encoding, decoded back to \
                 text. `truncate_to_tokens(text, 5, 'gpt-4o')`.",
                &[
                    "truncate",
                    "truncate to tokens for model",
                    "clip tokens",
                    "token budget",
                    "gpt-4o",
                    "limit tokens",
                    "context window",
                    "o200k_base",
                ],
                "shape",
                );
                tags.push(("vgi.example_queries".into(), truncate_examples_json()));
                tags
            },
            ..Default::default()
        }
    }

    fn argument_specs(&self) -> Vec<ArgSpec> {
        vec![
            ArgSpec::column("text", 0, "varchar", "The text to truncate."),
            ArgSpec::column(
                "n",
                1,
                "int32",
                "The maximum number of leading tokens to keep; the text is decoded back from these \
                 tokens. A value of 0 or less yields an empty string.",
            ),
            ArgSpec::column(
                "model",
                2,
                "varchar",
                "An LLM model name (e.g. 'gpt-4o') or a tiktoken encoding name (e.g. 'o200k_base') \
                 that selects which BPE encoding tokenizes the text before truncation. An unknown \
                 model yields NULL.",
            ),
        ]
    }

    fn on_bind(&self, _params: &BindParams) -> Result<BindResponse> {
        Ok(BindResponse::result(DataType::Utf8))
    }

    fn process(&self, params: &ProcessParams, batch: &RecordBatch) -> Result<RecordBatch> {
        let text = batch.column(0);
        let n = batch.column(1);
        let model = batch.column(2);
        let rows = batch.num_rows();
        let mut out = StringBuilder::new();
        for i in 0..rows {
            match (text_str(text, i)?, int_val(n, i)?, text_str(model, i)?) {
                (Some(t), Some(n), Some(m)) => match tiktoken::resolve(m) {
                    Some(enc) => {
                        let n = n.max(0) as usize;
                        out.append_value(tiktoken::truncate(t, n, enc));
                    }
                    None => out.append_null(), // unknown model → NULL
                },
                _ => out.append_null(),
            }
        }
        let arr: ArrayRef = Arc::new(out.finish());
        RecordBatch::try_new(params.output_schema.clone(), vec![arr])
            .map_err(|e| RpcError::runtime_error(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow_io::test_support::process_params;
    use arrow_array::cast::AsArray;
    use arrow_array::{Array, Int32Array, RecordBatch, StringArray};
    use arrow_schema::{Field, Schema};
    use vgi::arguments::Arguments;

    const TEXT: &str = "The quick brown fox jumps over the lazy dog.";

    fn run_trunc(texts: &[Option<&str>], ns: &[Option<i32>]) -> ArrayRef {
        let t: ArrayRef = Arc::new(StringArray::from(texts.to_vec()));
        let n: ArrayRef = Arc::new(Int32Array::from(ns.to_vec()));
        let schema = Arc::new(Schema::new(vec![
            Field::new("text", DataType::Utf8, true),
            Field::new("n", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![t, n]).unwrap();
        let bind = BindParams {
            input_schema: Some(schema),
            ..Default::default()
        };
        let bound = TruncateToTokens.on_bind(&bind).unwrap();
        let params = process_params(bound.output_schema, Arguments::default());
        TruncateToTokens
            .process(&params, &batch)
            .unwrap()
            .column(0)
            .clone()
    }

    #[test]
    fn truncate_prefix_and_round_trip() {
        let out = run_trunc(
            &[Some(TEXT), Some(TEXT), Some(TEXT), None],
            &[Some(3), Some(0), Some(1000), Some(3)],
        );
        let s = out.as_string::<i32>();
        // First 3 tokens, exactly.
        let three = s.value(0);
        assert_eq!(tiktoken::count(three, Encoding::Cl100kBase), 3);
        assert!(TEXT.starts_with(three));
        // n == 0 -> empty.
        assert_eq!(s.value(1), "");
        // n >= count -> whole text round-trips.
        assert_eq!(s.value(2), TEXT);
        // NULL text -> NULL.
        assert!(out.is_null(3));
    }
}
