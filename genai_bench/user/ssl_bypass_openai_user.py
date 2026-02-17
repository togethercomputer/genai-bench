"""OpenAI-compatible user with SSL certificate bypass.

Extends OpenAIUser with:
1. SSL bypass (verify=False) for internal endpoints with self-signed certs
2. Per-request params (messages, temperature, max_tokens) extracted from
   JSON-serialized request objects (e.g. SageMaker data capture format)
3. Optional authentication (supports unauthenticated endpoints)

Usage:
    genai-bench benchmark \
        --api-backend ssl-bypass-vllm \
        --api-base https://internal-endpoint:443 \
        --api-model-name <your-model-name> \
        --task text-to-text \
        --model-tokenizer <tokenizer>
"""

import json
import time
from typing import Any, Callable, Dict, List, Optional, Union

import requests
import urllib3
from locust import task

from genai_bench.logging import init_logger
from genai_bench.protocol import (
    UserChatRequest,
    UserImageChatRequest,
    UserResponse,
)
from genai_bench.user.openai_user import OpenAIUser

# Suppress InsecureRequestWarning from urllib3 when using verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = init_logger(__name__)


class SSLBypassVLLMUser(OpenAIUser):
    """OpenAI-compatible user with SSL bypass and per-request param support.

    This backend is useful for internal endpoints that:
    - Use self-signed SSL certificates
    - May not require authentication
    - Serve requests in OpenAI-compatible format (vLLM, SGLang, etc.)
    """

    BACKEND_NAME = "ssl-bypass-vllm"

    def on_start(self) -> None:
        if not self.host:
            raise ValueError("API base must be set.")

        # Support both authenticated and unauthenticated endpoints
        if self.auth_provider:
            try:
                auth_headers = self.auth_provider.get_headers()
                self.headers = {
                    **auth_headers,
                    "Content-Type": "application/json",
                }
            except Exception:
                # Fall back to unauthenticated if auth fails
                logger.warning(
                    "Auth provider failed to get headers, "
                    "proceeding without authentication."
                )
                self.headers = {"Content-Type": "application/json"}
        else:
            self.headers = {"Content-Type": "application/json"}

        self.api_backend = getattr(self, "api_backend", "vllm")
        logger.info(f"SSLBypassVLLMUser initialized for {self.host}")

    @task
    def chat(self):
        """Chat completion with per-request param extraction.

        If the prompt is a JSON-serialized request object (e.g. from SageMaker
        data capture), extracts messages, temperature, and max_tokens directly.
        Otherwise, falls back to the standard messages format.

        Expected JSON prompt format::

            {"messages": [...], "temperature": 0.0, "max_tokens": 100}
        """
        endpoint = "/v1/chat/completions"
        user_request = self.sample()

        if not isinstance(user_request, UserChatRequest):
            raise AttributeError(
                f"user_request should be of type "
                f"UserChatRequest for SSLBypassVLLMUser.chat, got "
                f"{type(user_request)}"
            )

        if isinstance(user_request, UserImageChatRequest):
            text_content = [{"type": "text", "text": user_request.prompt}]
            image_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": image},
                }
                for image in user_request.image_content
            ]
            content = text_content + image_content
        else:
            content = user_request.prompt

        # Try to parse prompt as a full request object (e.g. SageMaker
        # data capture format) to extract per-request params
        per_request_params: Dict[str, Any] = {}
        messages: Optional[List[Dict[str, Any]]] = None

        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "messages" in parsed:
                    # Full request object:
                    # {"messages": [...], "temperature": ..., "max_tokens": ...}
                    messages = parsed["messages"]
                    if parsed.get("temperature") is not None:
                        per_request_params["temperature"] = parsed["temperature"]
                    if parsed.get("max_tokens") is not None:
                        per_request_params["max_tokens"] = parsed["max_tokens"]
                elif isinstance(parsed, list) and parsed and "role" in parsed[0]:
                    # Plain messages array (backward compat)
                    messages = parsed
            except (json.JSONDecodeError, KeyError, IndexError):
                pass

        if messages is None:
            messages = self._build_messages(user_request, content)

        # Filter out keys that shouldn't be spread into payload
        filtered_params = {
            k: v
            for k, v in user_request.additional_request_params.items()
            if k not in {"system_message", "chat_history", "messages", "temperature"}
        }

        payload = {
            "model": user_request.model,
            "messages": messages,
            "max_tokens": per_request_params.get(
                "max_tokens", user_request.max_tokens
            ),
            "temperature": per_request_params.get(
                "temperature",
                user_request.additional_request_params.get("temperature", 0.0),
            ),
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
            **filtered_params,
        }

        # Conditionally add ignore_eos for vLLM and SGLang backends
        if self.api_backend in ("vllm", "sglang"):
            payload.setdefault("ignore_eos", bool(user_request.max_tokens))
        else:
            payload.pop("ignore_eos", None)

        self.send_request(
            True,
            endpoint,
            payload,
            self.parse_chat_response,
            user_request.num_prefill_tokens,
        )

    def send_request(
        self,
        stream: bool,
        endpoint: str,
        payload: Dict[str, Any],
        parse_strategy: Callable[..., UserResponse],
        num_prefill_tokens: Optional[int] = None,
    ) -> UserResponse:
        """Same as OpenAIUser.send_request but with verify=False for SSL bypass."""
        response = None

        try:
            start_time = time.monotonic()
            response = requests.post(
                url=f"{self.host}{endpoint}",
                json=payload,
                stream=stream,
                headers=self.headers,
                verify=False,
            )
            non_stream_post_end_time = time.monotonic()

            if response.status_code == 200:
                metrics_response = parse_strategy(
                    response,
                    start_time,
                    num_prefill_tokens,
                    non_stream_post_end_time,
                )
            else:
                metrics_response = UserResponse(
                    status_code=response.status_code,
                    error_message=response.text,
                )
        except requests.exceptions.ConnectionError as e:
            metrics_response = UserResponse(
                status_code=503, error_message=f"Connection error: {e}"
            )
        except requests.exceptions.Timeout as e:
            metrics_response = UserResponse(
                status_code=408, error_message=f"Request timed out: {e}"
            )
        except requests.exceptions.RequestException as e:
            metrics_response = UserResponse(
                status_code=500,
                error_message=str(e),
            )
        finally:
            if response is not None:
                response.close()

        self.collect_metrics(metrics_response, endpoint)
        return metrics_response

    @staticmethod
    def _build_messages(
        user_request: UserChatRequest,
        content: Union[str, list],
    ) -> List[Dict[str, Any]]:
        """Build the messages array from a user request and content.

        Incorporates system_message and chat_history from
        additional_request_params when available.

        Args:
            user_request: The sampled user request.
            content: The message content (string or multimodal list).

        Returns:
            A list of message dicts in OpenAI chat format.
        """
        messages: List[Dict[str, Any]] = []

        # Add system message if provided
        system_message = user_request.additional_request_params.get("system_message")
        if system_message:
            messages.append({"role": "system", "content": system_message})

        # Add conversation history if provided
        chat_history = user_request.additional_request_params.get("chat_history", [])
        for msg in chat_history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append(msg)

        # Add current user message
        messages.append({"role": "user", "content": content})

        return messages
