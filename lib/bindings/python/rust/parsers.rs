// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers::ReasoningParser;
use dynamo_parsers::reasoning::{ReasoningParserType, get_available_reasoning_parsers};
use dynamo_parsers::tool_calling::ToolDefinition;
use dynamo_parsers::tool_calling::parsers::{
    detect_and_parse_tool_call_with_recovery, get_available_tool_parsers,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::Value;

/// Get list of available tool parser names
#[pyfunction]
pub fn get_tool_parser_names() -> Vec<&'static str> {
    get_available_tool_parsers()
}

/// Get list of available reasoning parser names
#[pyfunction]
pub fn get_reasoning_parser_names() -> Vec<&'static str> {
    get_available_reasoning_parsers()
}

/// Parse tool calls from a model output string using the specified parser.
///
/// Uses the finalize / non-streaming aggregate path
/// (`detect_and_parse_tool_call_with_recovery`) so this binding mirrors what
/// Dynamo emits at end-of-response, including EOF-recovery for missing
/// end-token / truncated-JSON inputs. The streaming-safe variant (recovery
/// disabled) is intentionally NOT exposed here — it would compare the wrong
/// Dynamo behavior for batch-shaped fixtures (e.g. PARSER.batch.5).
///
/// Args:
///     parser_name: Parser name (e.g. "kimi_k25"). Empty string falls back to default.
///     message:     Model output text to parse.
///     tools_json:  Optional JSON-serialized list of tool definitions in the form
///                  `[{"name": "...", "parameters": {...}}, ...]`. Used by parsers
///                  that need schema-aware coercion (e.g. XML family).
///
/// Returns (awaited):
///     JSON-serialized string `{"calls": [...], "normal_text": str | null}` where
///     each entry in `calls` is `{"id", "type", "function": {"name", "arguments"}}`
///     and `arguments` is a JSON-serialized string (matching the parser's wire output).
///
/// Raises:
///     ValueError on parser failure or malformed `tools_json`.
#[pyfunction]
#[pyo3(signature = (parser_name, message, tools_json=None))]
pub fn parse_tool_call<'py>(
    py: Python<'py>,
    parser_name: String,
    message: String,
    tools_json: Option<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let tools = parse_tools_json(tools_json.as_deref())?;
    let parser_str = if parser_name.is_empty() {
        None
    } else {
        Some(parser_name)
    };

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let (calls, normal_text) = detect_and_parse_tool_call_with_recovery(
            &message,
            parser_str.as_deref(),
            tools.as_deref(),
        )
        .await
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;

        let result = serde_json::json!({
            "calls": calls,
            "normal_text": normal_text,
        });
        Ok(result.to_string())
    })
}

/// Parse reasoning from a complete model output string using the specified parser.
///
/// Args:
///     parser_name: Parser name (e.g. "qwen3"). Empty string falls back to the
///                  default reasoning parser.
///     message:     Model output text to parse.
///     token_ids:   Optional token IDs for parsers that need token-level markers.
///     in_reasoning:
///                  Start the parser in reasoning mode when the chat template
///                  already injected the opening reasoning marker.
///
/// Returns:
///     JSON-serialized string `{"reasoning_text": str, "normal_text": str}`.
#[pyfunction]
#[pyo3(signature = (parser_name, message, token_ids=None, in_reasoning=false))]
pub fn parse_reasoning(
    parser_name: String,
    message: String,
    token_ids: Option<Vec<u32>>,
    in_reasoning: bool,
) -> PyResult<String> {
    let mut parser = ReasoningParserType::get_reasoning_parser_from_name(&parser_name);
    if in_reasoning {
        parser.set_in_reasoning(true);
    }

    let token_ids = token_ids.unwrap_or_default();
    let result = parser.detect_and_parse_reasoning(&message, &token_ids);
    let out = serde_json::json!({
        "reasoning_text": result.reasoning_text,
        "normal_text": result.normal_text,
    });
    Ok(out.to_string())
}

/// Parse reasoning from streaming chunks using one stateful parser instance.
///
/// Args:
///     parser_name: Parser name (e.g. "qwen3"). Empty string falls back to the
///                  default reasoning parser.
///     chunks:      Model output text chunks in stream order.
///     token_chunks:
///                  Optional token ID chunks aligned 1:1 with `chunks`.
///     in_reasoning:
///                  Start the parser in reasoning mode when the chat template
///                  already injected the opening reasoning marker.
///
/// Returns:
///     JSON-serialized string with accumulated `reasoning_text` and `normal_text`.
#[pyfunction]
#[pyo3(signature = (parser_name, chunks, token_chunks=None, in_reasoning=false))]
pub fn parse_reasoning_stream(
    parser_name: String,
    chunks: Vec<String>,
    token_chunks: Option<Vec<Vec<u32>>>,
    in_reasoning: bool,
) -> PyResult<String> {
    if let Some(ref token_chunks) = token_chunks
        && token_chunks.len() != chunks.len()
    {
        return Err(PyValueError::new_err(format!(
            "token_chunks length ({}) must match chunks length ({})",
            token_chunks.len(),
            chunks.len()
        )));
    }

    let mut parser = ReasoningParserType::get_reasoning_parser_from_name(&parser_name);
    if in_reasoning {
        parser.set_in_reasoning(true);
    }

    let mut reasoning_text = String::new();
    let mut normal_text = String::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let token_ids = token_chunks
            .as_ref()
            .map(|chunks| chunks[i].as_slice())
            .unwrap_or(&[]);
        let result = parser.parse_reasoning_streaming_incremental(chunk, token_ids);
        reasoning_text.push_str(&result.reasoning_text);
        normal_text.push_str(&result.normal_text);
    }

    let out = serde_json::json!({
        "reasoning_text": reasoning_text,
        "normal_text": normal_text,
    });
    Ok(out.to_string())
}

/// Convert OpenAI-style or flat tools JSON into `Vec<ToolDefinition>`.
///
/// Accepts either of these shapes per element:
/// - `{"name": "fn", "parameters": {...}}`                                       (flat)
/// - `{"type": "function", "function": {"name": "fn", "parameters": {...}}}`    (OpenAI)
fn parse_tools_json(tools_json: Option<&str>) -> PyResult<Option<Vec<ToolDefinition>>> {
    let Some(raw) = tools_json else {
        return Ok(None);
    };
    let parsed: Value = serde_json::from_str(raw)
        .map_err(|e| PyValueError::new_err(format!("invalid tools_json: {e}")))?;
    let arr = parsed
        .as_array()
        .ok_or_else(|| PyValueError::new_err("tools_json must be a JSON array"))?;

    let mut defs = Vec::with_capacity(arr.len());
    for (i, t) in arr.iter().enumerate() {
        // OpenAI wraps the schema in `function`; fall back to flat shape.
        let inner = t.get("function").unwrap_or(t);
        let name = inner
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "tools_json[{i}] requires a string `name` field (or `function.name`)"
                ))
            })?
            .to_string();
        let parameters = inner.get("parameters").cloned();
        defs.push(ToolDefinition { name, parameters });
    }
    Ok(Some(defs))
}

/// Add parsers module functions to the Python module
pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_tool_parser_names, m)?)?;
    m.add_function(wrap_pyfunction!(get_reasoning_parser_names, m)?)?;
    m.add_function(wrap_pyfunction!(parse_tool_call, m)?)?;
    m.add_function(wrap_pyfunction!(parse_reasoning, m)?)?;
    m.add_function(wrap_pyfunction!(parse_reasoning_stream, m)?)?;
    Ok(())
}
