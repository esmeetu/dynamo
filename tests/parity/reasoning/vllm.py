# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reasoning parity wrapper for vLLM's reasoning parser classes."""

from __future__ import annotations

from typing import Any

from vllm.reasoning import ReasoningParserManager

from tests.parity.common import ReasoningResult

_FAMILY_TO_VLLM_REASONING = {
    "deepseek_v4": "deepseek_v4",
    "gemma4": "gemma4",
    "mistral": "mistral",
    "qwen3": "qwen3",
}


class _StubTokenizer:
    all_special_tokens: tuple[str, ...] = ()

    def encode(self, text: str, *args: Any, **kwargs: Any) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int], *args: Any, **kwargs: Any) -> str:
        return "".join(chr(t) for t in token_ids)


def _make_parser(parser_name: str, fixture: dict[str, Any]) -> Any:
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    tokenizer = _StubTokenizer()
    chat_template_kwargs = fixture.get("chat_template_kwargs", {})

    attempts = (
        lambda: parser_cls(tokenizer, chat_template_kwargs=chat_template_kwargs),
        lambda: parser_cls(tokenizer),
        lambda: parser_cls(),
    )
    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as e:
            last_error = e
    raise TypeError(f"could not initialize vLLM reasoning parser: {last_error}")


def _message_field(message: Any, *names: str) -> str:
    if message is None:
        return ""
    for name in names:
        if isinstance(message, dict):
            value = message.get(name)
        else:
            value = getattr(message, name, None)
        if value:
            return value
    return ""


def _run_stream(
    parser: Any, fixture: dict[str, Any], chunks: list[str]
) -> ReasoningResult:
    token_chunks = fixture.get("token_chunks")
    previous_text = ""
    previous_token_ids: list[int] = []
    reasoning_text = ""
    normal_text = ""
    reasoning_done = False
    tokenizer = _StubTokenizer()

    for i, chunk in enumerate(chunks):
        delta_token_ids = token_chunks[i] if token_chunks else tokenizer.encode(chunk)
        current_text = previous_text + chunk
        current_token_ids = previous_token_ids + delta_token_ids

        if reasoning_done:
            normal_text += chunk
        else:
            message = parser.extract_reasoning_streaming(
                previous_text,
                current_text,
                chunk,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
            )
            reasoning_text += _message_field(message, "reasoning", "reasoning_content")
            normal_text += _message_field(message, "content")
            if parser.is_reasoning_end_streaming(current_token_ids, delta_token_ids):
                reasoning_done = True

        previous_text = current_text
        previous_token_ids = current_token_ids

    return ReasoningResult(reasoning_text=reasoning_text, normal_text=normal_text)


def parse(
    parser_family: str,
    fixture: dict[str, Any],
    mode: str,
) -> ReasoningResult:
    parser_name = fixture.get("vllm_parser") or _FAMILY_TO_VLLM_REASONING.get(
        parser_family
    )
    if not parser_name:
        return ReasoningResult(
            error=f"UNAVAILABLE: vLLM has no reasoning parser for family={parser_family!r}"
        )

    try:
        parser = _make_parser(parser_name, fixture)
        if mode == "stream":
            chunks = fixture["chunks"]
        else:
            chunks = [fixture["model_text"]]
        return _run_stream(parser, fixture, chunks)
    except Exception as e:
        return ReasoningResult(error=f"{type(e).__name__}: {e}")
