//! `tokenize(text) -> INTEGER[]` (default cl100k_base) and
//! `tokenize(text, model) -> INTEGER[]` — the BPE token ids of `text`.
//!
//! NULL text → NULL list. Empty text → empty list. Unknown model → NULL list
//! (consistent with `count_tokens`).

use arrow_array::builder::Int32Builder;
use arrow_array::builder::ListBuilder;
use arrow_array::{ArrayRef, RecordBatch};
use vgi::{
    ArgSpec, BindParams, BindResponse, FunctionExample, FunctionMetadata, ProcessParams,
    ScalarFunction,
};
use vgi_rpc::{Result, RpcError};

use crate::arrow_io::{finish_list_int, list_int_builder, list_int_type, text_str};
use crate::tiktoken::{self, Encoding};

/// `tokenize(text)` — token ids under the default encoding (cl100k_base).
pub struct Tokenize;

impl ScalarFunction for Tokenize {
    fn name(&self) -> &str {
        "tokenize"
    }

    fn metadata(&self) -> FunctionMetadata {
        FunctionMetadata {
            description: "Tokenize text to BPE token ids under the default encoding \
                          (cl100k_base) as INTEGER[]. Empty -> []; NULL -> NULL"
                .into(),
            return_type: Some(list_int_type()),
            examples: vec![FunctionExample {
                sql: "SELECT tiktoken.main.tokenize('tiktoken is great!');".into(),
                description: "Get the cl100k_base BPE token ids for a string as an INTEGER[]."
                    .into(),
                expected_output: None,
            }],
            tags: crate::meta::object_tags(
                "Tokenize To Ids (Default Encoding)",
                "Tokenize text into its BPE token ids under the default encoding (cl100k_base) and \
                 return them as an INTEGER[]. Empty text -> []; NULL -> NULL. Use to inspect how \
                 a string is split into tokens.",
                "Tokenize text to cl100k_base BPE token ids as INTEGER[]. \
                 `tokenize('hello world')` -> `[15339, 1917]`.",
                "tokenize, token ids, bpe ids, encode text, tokens array, cl100k_base, token \
                 list, llm tokens",
                "scalar/tokenize.rs",
            ),
            ..Default::default()
        }
    }

    fn argument_specs(&self) -> Vec<ArgSpec> {
        vec![ArgSpec::any_column("text", 0, "Text to tokenize (VARCHAR)")]
    }

    fn on_bind(&self, _params: &BindParams) -> Result<BindResponse> {
        Ok(BindResponse::result(list_int_type()))
    }

    fn process(&self, params: &ProcessParams, batch: &RecordBatch) -> Result<RecordBatch> {
        let col = batch.column(0);
        let rows = batch.num_rows();
        let mut builder: ListBuilder<Int32Builder> = list_int_builder();
        for i in 0..rows {
            match text_str(col, i)? {
                None => builder.append_null(),
                Some(text) => {
                    for id in tiktoken::tokenize(text, Encoding::default_encoding()) {
                        builder.values().append_value(id as i32);
                    }
                    builder.append(true);
                }
            }
        }
        let arr: ArrayRef = finish_list_int(builder);
        debug_assert_eq!(arr.data_type(), &list_int_type());
        RecordBatch::try_new(params.output_schema.clone(), vec![arr])
            .map_err(|e| RpcError::runtime_error(e.to_string()))
    }
}

/// `tokenize(text, model)` — token ids under the encoding for `model`.
pub struct TokenizeModel;

impl ScalarFunction for TokenizeModel {
    fn name(&self) -> &str {
        "tokenize"
    }

    fn metadata(&self) -> FunctionMetadata {
        FunctionMetadata {
            description: "Tokenize text to BPE token ids using the encoding for the given model \
                          name (e.g. 'gpt-4o') as INTEGER[]. Unknown model -> NULL; empty -> []"
                .into(),
            return_type: Some(list_int_type()),
            examples: vec![FunctionExample {
                sql: "SELECT tiktoken.main.tokenize('hello world', 'gpt-4o');".into(),
                description:
                    "Tokenize text using a specific model's encoding (gpt-4o uses o200k_base)."
                        .into(),
                expected_output: None,
            }],
            tags: crate::meta::object_tags(
                "Tokenize To Ids For Model",
                "Tokenize text into its BPE token ids using the encoding for the given model name \
                 (e.g. 'gpt-4o') and return them as an INTEGER[]. Unknown model -> NULL; empty \
                 text -> []. Use to inspect how a specific model splits a string into tokens.",
                "Tokenize text to BPE token ids using a model's encoding as INTEGER[]. \
                 `tokenize('hi', 'gpt-4o')` uses o200k_base.",
                "tokenize, token ids for model, bpe ids, encode text, gpt-4o tokens, tokens \
                 array, o200k_base, llm tokens",
                "scalar/tokenize.rs",
            ),
            ..Default::default()
        }
    }

    fn argument_specs(&self) -> Vec<ArgSpec> {
        vec![
            ArgSpec::any_column("text", 0, "Text to tokenize (VARCHAR)"),
            ArgSpec::any_column(
                "model",
                1,
                "Model or encoding name, e.g. 'gpt-4o' (VARCHAR)",
            ),
        ]
    }

    fn on_bind(&self, _params: &BindParams) -> Result<BindResponse> {
        Ok(BindResponse::result(list_int_type()))
    }

    fn process(&self, params: &ProcessParams, batch: &RecordBatch) -> Result<RecordBatch> {
        let text = batch.column(0);
        let model = batch.column(1);
        let rows = batch.num_rows();
        let mut builder: ListBuilder<Int32Builder> = list_int_builder();
        for i in 0..rows {
            match (text_str(text, i)?, text_str(model, i)?) {
                (Some(t), Some(m)) => match tiktoken::resolve(m) {
                    Some(enc) => {
                        for id in tiktoken::tokenize(t, enc) {
                            builder.values().append_value(id as i32);
                        }
                        builder.append(true);
                    }
                    None => builder.append_null(), // unknown model → NULL list
                },
                _ => builder.append_null(),
            }
        }
        let arr: ArrayRef = finish_list_int(builder);
        debug_assert_eq!(arr.data_type(), &list_int_type());
        RecordBatch::try_new(params.output_schema.clone(), vec![arr])
            .map_err(|e| RpcError::runtime_error(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow_io::test_support::{
        bound_type, int_list_row, process_params, run_scalar_text,
    };
    use arrow_array::{Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;
    use vgi::arguments::Arguments;

    #[test]
    fn tokenize_default_round_trips() {
        assert_eq!(bound_type(&Tokenize), list_int_type());
        let out = run_scalar_text(
            &Tokenize,
            &[Some("tiktoken is great!"), Some(""), None],
            Arguments::default(),
        )
        .unwrap();
        let ids = int_list_row(&out, 0);
        // Matches count_tokens, and decodes back to the input.
        assert_eq!(
            ids.len(),
            tiktoken::count("tiktoken is great!", Encoding::Cl100kBase)
        );
        let as_u32: Vec<u32> = ids.iter().map(|i| *i as u32).collect();
        assert_eq!(
            tiktoken::decode(&as_u32, Encoding::Cl100kBase),
            "tiktoken is great!"
        );
        assert_eq!(int_list_row(&out, 1), Vec::<i32>::new(), "empty -> []");
        assert!(out.is_null(2), "NULL -> NULL");
    }

    #[test]
    fn tokenize_with_model() {
        let t: ArrayRef = Arc::new(StringArray::from(vec![Some("hello world"), Some("x")]));
        let m: ArrayRef = Arc::new(StringArray::from(vec![Some("gpt-4o"), Some("bogus")]));
        let schema = Arc::new(Schema::new(vec![
            Field::new("text", DataType::Utf8, true),
            Field::new("model", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![t, m]).unwrap();
        let bind = BindParams {
            input_schema: Some(schema),
            ..Default::default()
        };
        let bound = TokenizeModel.on_bind(&bind).unwrap();
        let params = process_params(bound.output_schema, Arguments::default());
        let out = TokenizeModel
            .process(&params, &batch)
            .unwrap()
            .column(0)
            .clone();
        assert_eq!(
            int_list_row(&out, 0).len(),
            2,
            "o200k 'hello world' = 2 tokens"
        );
        assert!(out.is_null(1), "unknown model -> NULL");
    }
}
