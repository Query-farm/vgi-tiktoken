//! `encoding_for_model(model VARCHAR) -> VARCHAR` — map a model name to its
//! tiktoken encoding name (e.g. `'gpt-4o'` → `'o200k_base'`). An **unknown
//! model** yields **NULL** (documented: treat as missing, never an error). NULL
//! in → NULL out.

use std::sync::Arc;

use arrow_array::builder::StringBuilder;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::DataType;
use vgi::{ArgSpec, BindParams, BindResponse, FunctionMetadata, ProcessParams, ScalarFunction};
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
            ..Default::default()
        }
    }

    fn argument_specs(&self) -> Vec<ArgSpec> {
        vec![ArgSpec::any_column(
            "model",
            0,
            "Model name, e.g. 'gpt-4o' (VARCHAR)",
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
