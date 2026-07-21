//! `count_tokens(text) -> INTEGER` (default cl100k_base) and
//! `count_tokens(text, model) -> INTEGER` (encoding chosen by model name).
//!
//! Two arity overloads share the name `count_tokens`; VGI's overload resolver
//! picks by argument count. Empty text → 0. NULL text → NULL.
//!
//! ## Unknown-model policy
//!
//! An **unknown model** is treated as missing metadata → the row's result is
//! **NULL** (mirrors `encoding_for_model`, and how a NULL flows to NULL). This
//! tolerates dirty/unknown model strings in data without aborting a scan. The
//! caller can `WHERE encoding_for_model(model) IS NOT NULL` to find them.

use std::sync::Arc;

use arrow_array::builder::Int32Builder;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::DataType;
use vgi::{
    ArgSpec, BindParams, BindResponse, FunctionExample, FunctionMetadata, ProcessParams,
    ScalarFunction,
};
use vgi_rpc::{Result, RpcError};

use crate::arrow_io::text_str;
use crate::tiktoken::{self, Encoding};

/// The `vgi.example_queries` payload for `count_tokens`. Both arity overloads
/// carry the **same** described-example list: the vgi extension exposes overloads
/// as one `duckdb_functions()` row whose native `examples` column is the union of
/// every overload's `Meta.examples`, so the winning tag must describe every one
/// of those SQL strings (VGI515) — an overload-specific tag would leave the other
/// overload's native example without a description.
fn count_examples_json() -> String {
    crate::meta::example_queries_json(&[
        (
            "Count the GPT-4/3.5 (cl100k_base) tokens in a sentence.",
            "SELECT tiktoken.main.count_tokens('The quick brown fox jumps over the lazy dog.');",
        ),
        (
            "Count tokens with the encoding for a specific model (gpt-4o uses o200k_base) to budget a context window.",
            "SELECT tiktoken.main.count_tokens('Summarize this prompt for GPT-4o.', 'gpt-4o');",
        ),
    ])
}

/// `count_tokens(text)` — token count under the default encoding (cl100k_base).
pub struct CountTokens;

impl ScalarFunction for CountTokens {
    fn name(&self) -> &str {
        "count_tokens"
    }

    fn metadata(&self) -> FunctionMetadata {
        FunctionMetadata {
            description: "Count LLM tokens in text under the default encoding (cl100k_base, the \
                          GPT-4/3.5 tokenizer). Empty text -> 0"
                .into(),
            return_type: Some(DataType::Int32),
            examples: vec![FunctionExample {
                sql: "SELECT tiktoken.main.count_tokens('The quick brown fox jumps over the lazy dog.');".into(),
                description: "Count the GPT-4/3.5 (cl100k_base) tokens in a sentence.".into(),
                expected_output: None,
            }],
            tags: {
                let mut tags = crate::meta::object_tags(
                "Count Tokens (Default Encoding)",
                "Count the exact number of LLM tokens in text under the default encoding \
                 (cl100k_base, the GPT-4/3.5 tokenizer). Empty text -> 0; NULL -> NULL. Use to \
                 budget prompts and context windows or estimate API token cost.",
                "Count LLM tokens in text under the default cl100k_base encoding. \
                 `count_tokens('hello world')` -> 2.",
                &[
                    "count tokens",
                    "token count",
                    "num tokens",
                    "tokenize count",
                    "context window",
                    "prompt budget",
                    "cl100k_base",
                    "gpt-4",
                    "gpt-3.5",
                    "llm tokens",
                ],
                "count",
                );
                tags.push(("vgi.example_queries".into(), count_examples_json()));
                tags
            },
            ..Default::default()
        }
    }

    fn argument_specs(&self) -> Vec<ArgSpec> {
        vec![ArgSpec::column(
            "text",
            0,
            "varchar",
            "The text whose tokens are counted.",
        )]
    }

    fn on_bind(&self, _params: &BindParams) -> Result<BindResponse> {
        Ok(BindResponse::result(DataType::Int32))
    }

    fn process(&self, params: &ProcessParams, batch: &RecordBatch) -> Result<RecordBatch> {
        let col = batch.column(0);
        let rows = batch.num_rows();
        let mut out = Int32Builder::new();
        for i in 0..rows {
            match text_str(col, i)? {
                Some(text) => {
                    out.append_value(tiktoken::count(text, Encoding::default_encoding()) as i32)
                }
                None => out.append_null(),
            }
        }
        let arr: ArrayRef = Arc::new(out.finish());
        RecordBatch::try_new(params.output_schema.clone(), vec![arr])
            .map_err(|e| RpcError::runtime_error(e.to_string()))
    }
}

/// `count_tokens(text, model)` — token count under the encoding for `model`.
/// Unknown model → NULL.
pub struct CountTokensModel;

impl ScalarFunction for CountTokensModel {
    fn name(&self) -> &str {
        "count_tokens"
    }

    fn metadata(&self) -> FunctionMetadata {
        FunctionMetadata {
            description: "Count LLM tokens in text using the encoding for the given model name \
                          (e.g. 'gpt-4o'). Exact for OpenAI BPE families, a close proxy for \
                          others. Unknown model -> NULL; empty text -> 0"
                .into(),
            return_type: Some(DataType::Int32),
            examples: vec![FunctionExample {
                sql: "SELECT tiktoken.main.count_tokens('Summarize this prompt for GPT-4o.', 'gpt-4o');".into(),
                description: "Count tokens with the encoding for a specific model (gpt-4o uses o200k_base) to budget a context window.".into(),
                expected_output: None,
            }],
            tags: {
                let mut tags = crate::meta::object_tags(
                "Count Tokens For Model",
                "Count the exact number of LLM tokens in text using the encoding for the given \
                 model name (e.g. 'gpt-4o'). Exact for OpenAI BPE families. Unknown model -> \
                 NULL; empty text -> 0. Use to budget a specific model's context window.",
                "Count LLM tokens in text using a model's encoding. \
                 `count_tokens('hi', 'gpt-4o')` counts under o200k_base.",
                &[
                    "count tokens",
                    "token count for model",
                    "gpt-4o tokens",
                    "model token count",
                    "context window",
                    "prompt budget",
                    "o200k_base",
                    "llm tokens",
                ],
                "count",
                );
                tags.push(("vgi.example_queries".into(), count_examples_json()));
                tags
            },
            ..Default::default()
        }
    }

    fn argument_specs(&self) -> Vec<ArgSpec> {
        vec![
            ArgSpec::column("text", 0, "varchar", "The text whose tokens are counted."),
            ArgSpec::column(
                "model",
                1,
                "varchar",
                "An LLM model name (e.g. 'gpt-4o') or a tiktoken encoding name (e.g. 'o200k_base') \
                 that selects which BPE encoding is used to count tokens. An unknown model yields \
                 NULL.",
            ),
        ]
    }

    fn on_bind(&self, _params: &BindParams) -> Result<BindResponse> {
        Ok(BindResponse::result(DataType::Int32))
    }

    fn process(&self, params: &ProcessParams, batch: &RecordBatch) -> Result<RecordBatch> {
        let text = batch.column(0);
        let model = batch.column(1);
        let rows = batch.num_rows();
        let mut out = Int32Builder::new();
        for i in 0..rows {
            match (text_str(text, i)?, text_str(model, i)?) {
                (Some(t), Some(m)) => match tiktoken::resolve(m) {
                    Some(enc) => out.append_value(tiktoken::count(t, enc) as i32),
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
    use crate::arrow_io::test_support::{bound_type, process_params, run_scalar_text};
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int32Type;
    use arrow_array::{Array, RecordBatch, StringArray};
    use arrow_schema::{Field, Schema};
    use vgi::arguments::Arguments;

    #[test]
    fn count_default_hello_world_is_two() {
        assert_eq!(bound_type(&CountTokens), DataType::Int32);
        let out = run_scalar_text(
            &CountTokens,
            &[Some("hello world"), Some(""), None],
            Arguments::default(),
        )
        .unwrap();
        let v = out.as_primitive::<Int32Type>();
        assert_eq!(v.value(0), 2, "cl100k_base 'hello world' = 2 tokens");
        assert_eq!(v.value(1), 0, "empty -> 0");
        assert!(out.is_null(2), "NULL -> NULL");
    }

    fn run_count_model(texts: &[Option<&str>], models: &[Option<&str>]) -> ArrayRef {
        let t: ArrayRef = Arc::new(StringArray::from(texts.to_vec()));
        let m: ArrayRef = Arc::new(StringArray::from(models.to_vec()));
        let schema = Arc::new(Schema::new(vec![
            Field::new("text", DataType::Utf8, true),
            Field::new("model", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![t, m]).unwrap();
        let bind = BindParams {
            input_schema: Some(schema),
            ..Default::default()
        };
        let bound = CountTokensModel.on_bind(&bind).unwrap();
        let params = process_params(bound.output_schema, Arguments::default());
        CountTokensModel
            .process(&params, &batch)
            .unwrap()
            .column(0)
            .clone()
    }

    #[test]
    fn count_with_model() {
        let out = run_count_model(
            &[Some("hello world"), Some("hello world"), Some("x"), None],
            &[
                Some("gpt-4o"),
                Some("gpt-4"),
                Some("bogus-model"),
                Some("gpt-4"),
            ],
        );
        let v = out.as_primitive::<Int32Type>();
        // Both encodings tokenize "hello world" as 2 tokens.
        assert_eq!(v.value(0), 2, "gpt-4o (o200k) 'hello world' = 2");
        assert_eq!(v.value(1), 2, "gpt-4 (cl100k) 'hello world' = 2");
        assert!(out.is_null(2), "unknown model -> NULL");
        assert!(out.is_null(3), "NULL text -> NULL");
    }
}
