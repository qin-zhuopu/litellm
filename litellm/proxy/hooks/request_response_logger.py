# """
# RequestResponseLogger - Log all API requests and responses to JSONL files
#
# This custom logger creates 4 separate JSONL log files for each API call:
# 1. downstream_request.jsonl - Client request sent to LiteLLM Proxy
# 2. downstream_response.jsonl - Response returned to client
# 3. upstream_request.jsonl - Request sent to model provider
# 4. upstream_response.jsonl - Response from model provider
#
# Each log entry includes:
# - timestamp: ISO format timestamp
# - call_id: Unique identifier to correlate all 4 logs for a single request
# - Relevant request/response data
#
# Configuration via config.yaml:
#   litellm_settings:
#     request_response_logger_params:
#       log_dir: "/path/to/logs"
#
#   callbacks:
#     - request_response_logger
# """
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import ModelResponse
from litellm._logging import verbose_proxy_logger


class RequestResponseLogger(CustomLogger):
    """
    记录每次 API 调用的上下游请求/响应体到 JSONL 文件。

    每次调用产生 4 条日志：
    1. downstream_request - 客户端发送给 Proxy 的请求
    2. downstream_response - Proxy 返回给客户端的响应
    3. upstream_request - Proxy 发送给模型提供商的请求
    4. upstream_response - 模型提供商返回的响应
    """

    def __init__(
        self,
        log_dir: str = "/workspace/litellm/logs",
        turn_off_message_logging: bool = False,
    ):
        super().__init__(turn_off_message_logging=turn_off_message_logging)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 为每种日志类型创建单独的文件
        self.downstream_request_log = self.log_dir / "downstream_request.jsonl"
        self.downstream_response_log = self.log_dir / "downstream_response.jsonl"
        self.upstream_request_log = self.log_dir / "upstream_request.jsonl"
        self.upstream_response_log = self.log_dir / "upstream_response.jsonl"

        # 用于关联请求和响应的 call_id
        self._call_context: Dict[str, Dict] = {}

    @classmethod
    def from_config_yaml(cls, config: Dict[str, Any]) -> "RequestResponseLogger":
        """
        Initialize RequestResponseLogger from proxy config.yaml parameters.

        Args:
            config: Configuration dictionary from callback_specific_params or litellm_settings

        Returns:
            Configured RequestResponseLogger instance

        Example:
            From proxy_config.yaml:
                general_settings:
                  callback_specific_params:
                    request_response_logger:
                      log_dir: "/var/log/litellm"

                litellm_settings:
                  request_response_logger_params:
                    log_dir: "/var/log/litellm"

            Usage:
                logger = RequestResponseLogger.from_config_yaml(config)
        """
        log_dir = config.get("log_dir", "/workspace/litellm/logs")
        return cls(log_dir=log_dir)

    @staticmethod
    def initialize_from_proxy_config(
        litellm_settings: Dict[str, Any],
        callback_specific_params: Dict[str, Any],
    ) -> "RequestResponseLogger":
        """
        Static method to initialize RequestResponseLogger from proxy config.

        Used in callback_utils.py to simplify initialization logic.

        Args:
            litellm_settings: Dictionary containing litellm_settings from proxy_config.yaml
            callback_specific_params: Dictionary containing callback-specific parameters

        Returns:
            Configured RequestResponseLogger instance
        """
        # Get request_response_logger_params from litellm_settings or callback_specific_params
        logger_params: Dict[str, Any] = {}
        if "request_response_logger_params" in litellm_settings:
            logger_params = litellm_settings["request_response_logger_params"]
        elif "request_response_logger" in callback_specific_params:
            logger_params = callback_specific_params["request_response_logger"]

        return RequestResponseLogger.from_config_yaml(logger_params)

    def _write_jsonl(self, file_path: Path, data: dict):
        """Write to JSONL log file"""
        try:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            verbose_proxy_logger.error(f"[RequestResponseLogger] Failed to write to {file_path}: {e}")

    async def _async_write_jsonl(self, file_path: Path, data: dict):
        """Async write to JSONL log file"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._write_jsonl, file_path, data)

    def _get_call_id(self, data: dict) -> Optional[str]:
        """Extract call_id from request data"""
        return data.get("litellm_call_id") or data.get("x-litellm-call-id")

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> Optional[dict]:
        """
        Proxy 专用钩子：在调用前记录下游请求。
        """
        try:
            call_id = self._get_call_id(data)

            # 提取原始模型（用户请求的模型）
            original_model = data.get("original_model")
            request_data = self._sanitize_request(data)

            # 下游请求记录用户原始请求的模型
            if original_model:
                request_data["model"] = original_model  # 用原始模型覆盖被改的值

            # 记录下游请求
            downstream_request = {
                "timestamp": datetime.utcnow().isoformat(),
                "call_id": call_id,
                "call_type": call_type,
                "request": request_data,
            }
            await self._async_write_jsonl(self.downstream_request_log, downstream_request)

            # 保存上下文，用于后续关联
            if call_id:
                self._call_context[call_id] = {
                    "request_data": data,
                    "start_time": datetime.utcnow().isoformat(),
                }

            verbose_proxy_logger.debug(f"[RequestResponseLogger] Logged downstream request for call_id={call_id}")
        except Exception as e:
            verbose_proxy_logger.error(f"[RequestResponseLogger] Error in async_pre_call_hook: {e}")

        return data  # 不修改请求，直接返回

    def log_pre_api_call(self, model: str, messages: list, kwargs: dict):
        """
        同步钩子：记录上游请求（发送给模型提供商的请求）。
        注意：这是同步方法，会在代理模式下被调用。
        """
        try:
            call_id = kwargs.get("litellm_call_id")

            # 构建上游请求日志
            upstream_request = {
                "timestamp": datetime.utcnow().isoformat(),
                "call_id": call_id,
                "model": model,
                "messages_count": len(messages) if messages else 0,
                "kwargs": self._sanitize_kwargs(kwargs),
            }

            # 同步写入
            self._write_jsonl(self.upstream_request_log, upstream_request)

            verbose_proxy_logger.debug(f"[RequestResponseLogger] Logged upstream request for call_id={call_id}")
        except Exception as e:
            verbose_proxy_logger.error(f"[RequestResponseLogger] Error in log_pre_api_call: {e}")

    async def async_log_success_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ):
        """
        异步钩子：记录上游响应（模型提供商返回的响应）。
        这个钩子会收到实际的响应对象。
        """
        try:
            call_id = kwargs.get("litellm_call_id")

            # 构建上游响应日志
            upstream_response = {
                "timestamp": datetime.utcnow().isoformat(),
                "call_id": call_id,
                "model": kwargs.get("model"),
                "response": self._serialize_response(response_obj),
            }

            await self._async_write_jsonl(self.upstream_response_log, upstream_response)

            verbose_proxy_logger.debug(f"[RequestResponseLogger] Logged upstream response for call_id={call_id}")
        except Exception as e:
            verbose_proxy_logger.error(f"[RequestResponseLogger] Error in async_log_success_event: {e}")

    async def async_post_call_success_hook(
        self, data: dict, user_api_key_dict: Any, response: Any
    ) -> Any:
        """
        Proxy 专用钩子：在成功响应后记录下游响应。
        """
        try:
            call_id = self._get_call_id(data)

            # 记录下游响应
            downstream_response = {
                "timestamp": datetime.utcnow().isoformat(),
                "call_id": call_id,
                "response": self._serialize_response(response),
            }

            await self._async_write_jsonl(self.downstream_response_log, downstream_response)

            # 清理上下文
            if call_id and call_id in self._call_context:
                del self._call_context[call_id]

            verbose_proxy_logger.debug(f"[RequestResponseLogger] Logged downstream response for call_id={call_id}")
        except Exception as e:
            verbose_proxy_logger.error(f"[RequestResponseLogger] Error in async_post_call_success_hook: {e}")

        return response  # 不修改响应，直接返回

    def _sanitize_request(self, data: dict) -> dict:
        """
        Sanitize request data for logging.
        Remove sensitive fields and reduce size.
        """
        if not data:
            return {}

        # Make a shallow copy to avoid modifying original
        sanitized = {}

        # Copy safe fields
        safe_fields = [
            "model", "messages", "tools", "tool_choice", "temperature",
            "max_tokens", "top_p", "stream", "n", "stop", "presence_penalty",
            "frequency_penalty", "user", "function_call", "functions",
            "response_format", "seed", "tool_id"
        ]

        for field in safe_fields:
            if field in data:
                value = data[field]
                # Truncate messages if too long
                if field == "messages" and isinstance(value, list):
                    sanitized[field] = self._truncate_messages(value)
                else:
                    sanitized[field] = value

        # Add call_id if present
        if "litellm_call_id" in data:
            sanitized["litellm_call_id"] = data["litellm_call_id"]

        return sanitized

    def _sanitize_kwargs(self, kwargs: dict) -> dict:
        """Sanitize kwargs for logging"""
        if not kwargs:
            return {}

        sanitized = {}

        # Safe fields from kwargs
        safe_fields = [
            "model", "messages", "tools", "temperature", "max_tokens",
            "top_p", "stream", "litellm_call_id", "api_base", "headers"
        ]

        for field in safe_fields:
            if field in kwargs:
                value = kwargs[field]
                if field == "messages" and isinstance(value, list):
                    sanitized[field] = self._truncate_messages(value)
                elif field == "headers":
                    # Redact sensitive headers
                    sanitized[field] = self._sanitize_headers(value)
                else:
                    sanitized[field] = value

        return sanitized

    def _sanitize_headers(self, headers: Any) -> dict:
        """Redact sensitive headers"""
        if not headers:
            return {}

        sanitized = {}
        for key, value in headers.items() if isinstance(headers, dict) else []:
            key_lower = key.lower()
            # Redact sensitive headers
            if any(sensitive in key_lower for sensitive in ["authorization", "api-key", "x-api-key"]):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value

        return sanitized

    def _truncate_messages(self, messages: list, max_length: int = 1000) -> list:
        """Truncate message content to reduce log size"""
        if not messages:
            return []

        truncated = []
        for msg in messages:
            if isinstance(msg, dict):
                truncated_msg = {}
                for key, value in msg.items():
                    if key == "content" and isinstance(value, str):
                        # Truncate long content
                        if len(value) > max_length:
                            truncated_msg[key] = value[:max_length] + "...[truncated]"
                        else:
                            truncated_msg[key] = value
                    elif key == "content" and isinstance(value, list):
                        # Handle multimodal content
                        truncated_content = []
                        for item in value:
                            if isinstance(item, dict):
                                truncated_item = {}
                                for k, v in item.items():
                                    if k == "text" and isinstance(v, str) and len(v) > max_length:
                                        truncated_item[k] = v[:max_length] + "...[truncated]"
                                    elif k == "image_url" or k == "source":
                                        # Redact image URLs
                                        truncated_item[k] = "[IMAGE]"
                                    else:
                                        truncated_item[k] = v
                                truncated_content.append(truncated_item)
                            else:
                                truncated_content.append(item)
                        truncated_msg[key] = truncated_content
                    else:
                        truncated_msg[key] = value
                truncated.append(truncated_msg)
            else:
                truncated.append(msg)

        return truncated

    def _serialize_response(self, response: Any) -> dict:
        """将响应对象序列化为字典"""
        if response is None:
            return {}

        if isinstance(response, dict):
            return response

        if hasattr(response, "model_dump"):
            try:
                return response.model_dump()
            except Exception:
                pass

        if hasattr(response, "dict"):
            try:
                return response.dict()
            except Exception:
                pass

        # ModelResponse 对象
        if isinstance(response, ModelResponse):
            return {
                "id": response.id,
                "model": response.model,
                "choices": [
                    {
                        "index": c.index,
                        "message": {
                            "role": c.message.role,
                            "content": (
                                c.message.content[:1000] + "..."
                                if c.message.content and len(c.message.content) > 1000
                                else c.message.content
                            )
                        },
                        "finish_reason": c.finish_reason,
                    }
                    for c in response.choices
                ] if response.choices else [],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                    "completion_tokens": response.usage.completion_tokens if response.usage else None,
                    "total_tokens": response.usage.total_tokens if response.usage else None,
                } if response.usage else {},
            }

        # Return string representation for unknown types
        return {
            "__type__": str(type(response)),
            "__str__": str(response),
            "__repr__": repr(response)
        }
