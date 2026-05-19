# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import logging
import struct
from collections.abc import AsyncGenerator
from typing import Any, Dict, List, Optional, Union

import sglang as sgl

from dynamo._core import Context
from dynamo.common.utils.otel_tracing import build_trace_headers
from dynamo.sglang.args import Config
from dynamo.sglang.protocol import EmbeddingRequest
from dynamo.sglang.publisher import DynamoSglangPublisher
from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler


class EmbeddingWorkerHandler(BaseWorkerHandler):
    def __init__(
        self,
        engine: sgl.Engine,
        config: Config,
        publisher: Optional[DynamoSglangPublisher] = None,
        shutdown_event: Optional[asyncio.Event] = None,
    ):
        super().__init__(engine, config, publisher, None, shutdown_event)
        logging.info("Embedding worker handler initialized")

    def cleanup(self) -> None:
        super().cleanup()
        self.engine.shutdown()
        logging.info("Engine shutdown")

    async def generate(
        self, request: dict, context: Context
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate embeddings for the given input.

        Args:
            request: Embedding request dictionary.
            context: Context object for cancellation handling.
        """
        logging.debug(f"Embedding request: {request}")

        # Parse the embedding request - should only receive EmbeddingRequest format
        embedding_request = EmbeddingRequest(**request)

        # Handle different input types
        prompt: str | list[Any]
        if isinstance(embedding_request.input, str):
            prompt = embedding_request.input
        elif isinstance(embedding_request.input, list):
            prompt = embedding_request.input
        else:
            raise TypeError(f"Invalid input type: {type(embedding_request.input)}")

        trace_header = build_trace_headers(context) if self.enable_trace else None
        trace_id = context.trace_id

        result = await self.engine.async_encode(
            prompt=prompt,
            external_trace_header=trace_header,
            rid=trace_id,
        )

        # Transform the response to OpenAI format
        response = self._transform_response(
            result,
            embedding_request.model,
            dimensions=embedding_request.dimensions,
            encoding_format=embedding_request.encoding_format or "float",
        )
        yield response

    def _transform_response(
        self,
        ret: Any,
        model_name: str,
        dimensions: Optional[int] = None,
        encoding_format: str = "float",
    ) -> Dict[str, Any]:
        """Transform SGLang response to OpenAI embedding format.

        Applies the two optional OpenAI fields:
          * ``dimensions``  -- Matryoshka-style truncation (slice leading N).
          * ``encoding_format`` -- ``"float"`` (JSON array) or ``"base64"``
            (little-endian float32 packed bytes, base64-ascii-encoded).
        """
        if encoding_format not in ("float", "base64"):
            raise ValueError(
                f"Invalid encoding_format {encoding_format!r}; "
                "expected 'float' or 'base64'"
            )

        if not isinstance(ret, list):
            ret = [ret]

        embedding_objects = []
        prompt_tokens = 0

        for idx, ret_item in enumerate(ret):
            embedding: List[float] = list(ret_item["embedding"])
            if dimensions is not None:
                if dimensions < 1:
                    raise ValueError(f"dimensions must be >= 1, got {dimensions}")
                if dimensions > len(embedding):
                    raise ValueError(
                        f"dimensions={dimensions} exceeds model embedding "
                        f"dimension {len(embedding)}"
                    )
                embedding = embedding[:dimensions]

            embedding_value: Union[List[float], str]
            if encoding_format == "base64":
                packed = struct.pack(f"<{len(embedding)}f", *embedding)
                embedding_value = base64.b64encode(packed).decode("ascii")
            else:
                embedding_value = embedding

            embedding_objects.append(
                {
                    "object": "embedding",
                    "embedding": embedding_value,
                    "index": idx,
                }
            )
            prompt_tokens += ret_item.get("meta_info", {}).get("prompt_tokens", 0)

        return {
            "object": "list",
            "data": embedding_objects,
            "model": model_name,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            },
        }
