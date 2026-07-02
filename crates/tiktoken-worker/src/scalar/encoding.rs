//! `encoding_for_model(model VARCHAR) -> VARCHAR` — map a model name to its
//! tiktoken encoding name (e.g. `'gpt-4o'` → `'o200k_base'`). An **unknown
//! model** yields **NULL** (documented: treat as missing, never an error). NULL
//! in → NULL out.

use std::sync::Arc;

use arrow_array::builder::StringBuilder;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::DataType;
use vgi::{
    ArgSpec, BindParams, BindResponse, FunctionExample, FunctionMetadata, ProcessParams,
    ScalarFunction,
};
use vgi_rpc::{Result, RpcError};

use crate::arrow_io::text_str;
use crate::tiktoken;

pub struct EncodingForModel;

impl ScalarFunction for EncodingForModel {
    fn name(&self) -> &str {
        "encoding_for_model"
    }

    fn metadata(&self) -> FunctionMetadata {
        FunctionMetadata {
            description: "Map an LLM model name to its tiktoken encoding name \
                          (e.g. 'gpt-4o' -> 'o200k_base'); NULL if the model is unknown"
                .into(),
            return_type: Some(DataType::Utf8),
            examples: vec![FunctionExample {
                sql: "SELECT tiktoken.main.encoding_for_model('gpt-4o');".into(),
                description: "Look up the tiktoken encoding a model uses (gpt-4o -> 'o200k_base')."
                    .into(),
                expected_output: None,
            }],
            tags: crate::meta::object_tags(
                "Encoding For Model Name",
                "Map an LLM model name (e.g. 'gpt-4o', 'gpt-4', 'gpt-3.5-turbo') to the name of \
                 the tiktoken BPE encoding it uses (e.g. 'o200k_base', 'cl100k_base'). Returns \
                 NULL for an unknown model. Use to pick the right encoding before counting or \
                 tokenizing.",
                "Map a model name to its tiktoken encoding name; NULL if unknown. \
                 `encoding_for_model('gpt-4o')` -> `o200k_base`.",
                &[
                    "encoding",
                    "encoding for model",
                    "tiktoken encoding",
                    "model to encoding",
                    "cl100k_base",
                    "o200k_base",
                    "p50k_base",
                    "bpe",
                    "tokenizer name",
                ],
                "reference",
            ),
            ..Default::default()
        }
    }

    fn argument_specs(&self) -> Vec<ArgSpec> {
        vec![ArgSpec::column(
            "model",
            0,
            "varchar",
            "An LLM model name (e.g. 'gpt-4o') to map to the tiktoken encoding it uses. An unknown \
             model yields NULL.",
        )]
    }

    fn on_bind(&self, _params: &BindParams) -> Result<BindResponse> {
        Ok(BindResponse::result(DataType::Utf8))
    }

    fn process(&self, params: &ProcessParams, batch: &RecordBatch) -> Result<RecordBatch> {
        let col = batch.column(0);
        let rows = batch.num_rows();
        let mut out = StringBuilder::new();
        for i in 0..rows {
            match text_str(col, i)? {
                Some(model) => match tiktoken::encoding_name_for_model(model) {
                    Some(name) => out.append_value(name),
                    None => out.append_null(), // unknown model → NULL
                },
                None => out.append_null(),
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
    use crate::arrow_io::test_support::{bound_type, run_scalar_text};
    use arrow_array::cast::AsArray;
    use arrow_array::Array;
    use vgi::arguments::Arguments;

    #[test]
    fn binds_and_maps_models() {
        assert_eq!(bound_type(&EncodingForModel), DataType::Utf8);
        let out = run_scalar_text(
            &EncodingForModel,
            &[Some("gpt-4o"), Some("gpt-4"), Some("nope"), None],
            Arguments::default(),
        )
        .unwrap();
        let s = out.as_string::<i32>();
        assert_eq!(s.value(0), "o200k_base");
        assert_eq!(s.value(1), "cl100k_base");
        assert!(out.is_null(2), "unknown model → NULL");
        assert!(out.is_null(3), "NULL in → NULL out");
    }
}
