//! Small Arrow helpers shared across the scalar functions: reading VARCHAR and
//! integer input cells, plus the `LIST(INT)` / `LIST(VARCHAR)` output builders
//! and their declared `DataType`s (which MUST match between `on_bind` and
//! `process` or DuckDB rejects the batch). The in-process test harness below
//! drives a `ScalarFunction` end-to-end without the RPC/IPC plumbing.

use std::sync::Arc;

use arrow_array::builder::{Int32Builder, ListBuilder, StringBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type,
    UInt64Type, UInt8Type,
};
use arrow_array::{Array, ArrayRef, ListArray};
use arrow_schema::{DataType, Field};
use vgi_rpc::{Result, RpcError};

/// Borrow the UTF-8 text of a VARCHAR cell at `row`, or `None` if null. Errors if
/// the column isn't a string type.
pub fn text_str(col: &ArrayRef, row: usize) -> Result<Option<&str>> {
    if col.is_null(row) {
        return Ok(None);
    }
    Ok(Some(match col.data_type() {
        DataType::Utf8 => col.as_string::<i32>().value(row),
        DataType::LargeUtf8 => col.as_string::<i64>().value(row),
        other => {
            return Err(RpcError::value_error(format!(
                "expected a VARCHAR (string) argument, got {other:?}"
            )))
        }
    }))
}

/// Read element `row` of a numeric column as `i64`, or `None` if null. Accepts
/// any of DuckDB's integer widths (an integer literal like
/// `truncate_to_tokens(t, 16)` may arrive as INT32 or INT64), and truncates a
/// float toward zero. Errors on a non-numeric column.
pub fn int_val(col: &ArrayRef, row: usize) -> Result<Option<i64>> {
    if col.is_null(row) {
        return Ok(None);
    }
    Ok(Some(match col.data_type() {
        DataType::Int64 => col.as_primitive::<Int64Type>().value(row),
        DataType::Int32 => col.as_primitive::<Int32Type>().value(row) as i64,
        DataType::Int16 => col.as_primitive::<Int16Type>().value(row) as i64,
        DataType::Int8 => col.as_primitive::<Int8Type>().value(row) as i64,
        DataType::UInt64 => col.as_primitive::<UInt64Type>().value(row) as i64,
        DataType::UInt32 => col.as_primitive::<UInt32Type>().value(row) as i64,
        DataType::UInt16 => col.as_primitive::<UInt16Type>().value(row) as i64,
        DataType::UInt8 => col.as_primitive::<UInt8Type>().value(row) as i64,
        DataType::Float64 => col.as_primitive::<Float64Type>().value(row) as i64,
        DataType::Float32 => col.as_primitive::<Float32Type>().value(row) as i64,
        other => {
            return Err(RpcError::value_error(format!(
                "expected an integer argument, got {other:?}"
            )))
        }
    }))
}

// ---------------------------------------------------------------------------
// LIST(INT) — token-id output for `tokenize`.
// ---------------------------------------------------------------------------

/// The Arrow `DataType` of a `LIST(INTEGER)` return. The element field is named
/// `item` and is nullable; this MUST match between `on_bind` and `process`.
pub fn list_int_type() -> DataType {
    DataType::List(Arc::new(Field::new("item", DataType::Int32, true)))
}

/// A `ListBuilder<Int32Builder>` configured with the `item`/nullable element
/// field so the built [`ListArray`]'s `DataType` equals [`list_int_type`].
pub fn list_int_builder() -> ListBuilder<Int32Builder> {
    let field = Arc::new(Field::new("item", DataType::Int32, true));
    ListBuilder::new(Int32Builder::new()).with_field(field)
}

/// Finish a `LIST(INT)` builder into an `ArrayRef`.
pub fn finish_list_int(mut b: ListBuilder<Int32Builder>) -> ArrayRef {
    let arr: ListArray = b.finish();
    Arc::new(arr)
}

// ---------------------------------------------------------------------------
// LIST(VARCHAR) — chunk output for `chunk_by_tokens`.
// ---------------------------------------------------------------------------

/// The Arrow `DataType` of a `LIST(VARCHAR)` return. Element field `item`,
/// nullable; MUST match between `on_bind` and `process`.
pub fn list_varchar_type() -> DataType {
    DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)))
}

/// A `ListBuilder<StringBuilder>` whose built [`ListArray`] equals
/// [`list_varchar_type`].
pub fn list_varchar_builder() -> ListBuilder<StringBuilder> {
    let field = Arc::new(Field::new("item", DataType::Utf8, true));
    ListBuilder::new(StringBuilder::new()).with_field(field)
}

/// Finish a `LIST(VARCHAR)` builder into an `ArrayRef`.
pub fn finish_list_varchar(mut b: ListBuilder<StringBuilder>) -> ArrayRef {
    let arr: ListArray = b.finish();
    Arc::new(arr)
}

/// Test-only helpers shared by the scalar Arrow-boundary unit tests. These let a
/// `#[cfg(test)]` block drive a `ScalarFunction` end to end in-process (build the
/// input `RecordBatch`, run `on_bind` + `process`, inspect the result) without
/// the RPC/IPC plumbing.
#[cfg(test)]
pub mod test_support {
    use std::sync::Arc;

    use arrow_array::builder::StringBuilder;
    use arrow_array::{ArrayRef, RecordBatch};
    use arrow_schema::{Field, Schema, SchemaRef};
    use vgi::arguments::Arguments;
    use vgi::{BindParams, ProcessParams, ScalarFunction};
    use vgi_rpc::Result;

    /// A single-column `Utf8` (VARCHAR) input batch. `None` entries become NULLs.
    pub fn text_batch(rows: &[Option<&str>]) -> RecordBatch {
        let mut b = StringBuilder::new();
        for r in rows {
            match r {
                Some(s) => b.append_value(s),
                None => b.append_null(),
            }
        }
        let arr: ArrayRef = Arc::new(b.finish());
        let schema = Arc::new(Schema::new(vec![Field::new(
            "text",
            arr.data_type().clone(),
            true,
        )]));
        RecordBatch::try_new(schema, vec![arr]).unwrap()
    }

    /// Build a `ProcessParams` carrying the given output schema and arguments.
    pub fn process_params(output_schema: SchemaRef, arguments: Arguments) -> ProcessParams {
        ProcessParams {
            substream_id: None,
            if_none_match: None,
            if_modified_since: None,
            output_schema,
            input_schema: None,
            execution_id: Vec::new(),
            init_opaque_data: Vec::new(),
            arguments,
            settings: Default::default(),
            secrets: Default::default(),
            auth_principal: None,
            projection_ids: None,
            pushdown_filters: None,
            join_keys: Vec::new(),
            storage: None,
            order_by_column: None,
            order_by_direction: None,
            order_by_null_order: None,
            order_by_limit: None,
            tablesample_percentage: None,
            tablesample_seed: None,
            attach_opaque_data: None,
            at_unit: None,
            at_value: None,
            copy_from: None,
        }
    }

    /// Run a scalar function over a prebuilt input batch: call `on_bind` to
    /// obtain the declared output schema, then `process`, returning the single
    /// result column. The `arguments` apply to both bind and process.
    pub fn run_scalar_on<F: ScalarFunction>(
        f: &F,
        batch: RecordBatch,
        arguments: Arguments,
    ) -> Result<ArrayRef> {
        let bind = BindParams {
            input_schema: Some(batch.schema()),
            arguments: arguments.clone(),
            ..Default::default()
        };
        let bound = f.on_bind(&bind)?;
        let params = process_params(bound.output_schema.clone(), arguments);
        let out = f.process(&params, &batch)?;
        Ok(out.column(0).clone())
    }

    /// Run a scalar over a single-column VARCHAR input batch.
    pub fn run_scalar_text<F: ScalarFunction>(
        f: &F,
        rows: &[Option<&str>],
        arguments: Arguments,
    ) -> Result<ArrayRef> {
        run_scalar_on(f, text_batch(rows), arguments)
    }

    /// The declared output `DataType` from `on_bind` for a scalar with no
    /// bind-time argument requirements.
    pub fn bound_type<F: ScalarFunction>(f: &F) -> arrow_schema::DataType {
        let bind = BindParams::default();
        let bound = f.on_bind(&bind).unwrap();
        bound.output_schema.field(0).data_type().clone()
    }

    /// Collect the INT elements at `row` of a `LIST(INT)` result into a `Vec<i32>`
    /// (panics if `row` is null).
    pub fn int_list_row(col: &ArrayRef, row: usize) -> Vec<i32> {
        use arrow_array::cast::AsArray;
        use arrow_array::types::Int32Type;
        let list = col.as_list::<i32>();
        let values = list.value(row);
        let v = values.as_primitive::<Int32Type>();
        (0..v.len()).map(|i| v.value(i)).collect()
    }

    /// Collect the VARCHAR elements at `row` of a `LIST(VARCHAR)` result into a
    /// `Vec<String>` (panics if `row` is null).
    pub fn varchar_list_row(col: &ArrayRef, row: usize) -> Vec<String> {
        use arrow_array::cast::AsArray;
        use arrow_array::Array;
        let list = col.as_list::<i32>();
        let values = list.value(row);
        let s = values.as_string::<i32>();
        (0..s.len()).map(|i| s.value(i).to_string()).collect()
    }
}
