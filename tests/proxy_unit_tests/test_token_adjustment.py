# Unit tests for token adjustment functions in common_request_processing.py


import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from litellm.proxy.common_request_processing import (
    _adjust_max_tokens_for_context_window,
    _estimate_input_tokens,
)


class TestTokenAdjustment:
    """Tests for context window token adjustment functions"""

    # ==========================================================================
    # Helper Functions
    # ==========================================================================

    def create_mock_deployment(self, model_name: str, actual_model: str):
        """Helper to create a mock Deployment object"""
        mock_params = MagicMock()
        mock_params.model = actual_model

        mock_deployment = MagicMock()
        mock_deployment.model_name = model_name
        mock_deployment.litellm_params = mock_params
        return mock_deployment

    def create_mock_router(self, deployment=None):
        """Helper to create a mock Router"""
        mock_router = MagicMock()
        mock_router.get_deployment_by_model_group_name = MagicMock(return_value=deployment)
        return mock_router

    # ==========================================================================
    # _estimate_input_tokens() Tests
    # ==========================================================================

    def test_estimate_empty_data(self):
        """Test with empty data dict returns minimum 1"""
        data = {}
        result = _estimate_input_tokens(data)
        assert result == 1

    def test_estimate_simple_message(self):
        """Test with simple message - 5 chars // 4 = 1"""
        data = {"messages": [{"role": "user", "content": "Hello"}]}
        result = _estimate_input_tokens(data)
        assert result == 1

    def test_estimate_long_message(self):
        """Test with long message - 400 chars // 4 = 100"""
        data = {"messages": [{"role": "user", "content": "x" * 400}]}
        result = _estimate_input_tokens(data)
        assert result == 100

    def test_estimate_multiple_messages(self):
        """Test with multiple messages of varying lengths"""
        data = {
            "messages": [
                {"role": "user", "content": "x" * 100},  # 25 tokens
                {"role": "assistant", "content": "y" * 200},  # 50 tokens
                {"role": "user", "content": "z" * 100},  # 25 tokens
            ]
        }
        result = _estimate_input_tokens(data)
        assert result == 100

    def test_estimate_multimodal_with_text(self):
        """Test multimodal content with text type"""
        data = {
            "messages": [
                {"content": [{"type": "text", "text": "hello"}]}
            ]
        }
        result = _estimate_input_tokens(data)
        assert result == 1  # 5 chars // 4

    def test_estimate_multimodal_with_image(self):
        """Test multimodal content with image - adds 1000 tokens"""
        data = {
            "messages": [
                {"content": [{"type": "image"}]}
            ]
        }
        result = _estimate_input_tokens(data)
        assert result == 1000

    def test_estimate_multimodal_with_text_and_image(self):
        """Test multimodal content with both text and image"""
        data = {
            "messages": [
                {"content": [{"type": "text", "text": "hello world"}, {"type": "image"}]}
            ]
        }
        result = _estimate_input_tokens(data)
        assert result == 1002  # 1000 for image + 2 for text

    def test_estimate_with_system_prompt(self):
        """Test estimation includes system prompt"""
        data = {
            "system": "system message " * 10,  # 145 chars -> 36 tokens
            "messages": [{"role": "user", "content": "hello"}]  # 5 chars -> 1 token
        }
        result = _estimate_input_tokens(data)
        assert result == 38  # 36 + 1 from messages, +1 from empty messages total_chars // 4

    def test_estimate_completion_api_format(self):
        """Test with completion API format (prompt instead of messages)"""
        data = {"prompt": "test prompt"}  # 11 chars // 4 = 2
        result = _estimate_input_tokens(data)
        assert result == 2

    def test_estimate_none_content_handling(self):
        """Test that None content is skipped gracefully"""
        data = {
            "messages": [
                {"role": "user", "content": None},
                {"role": "user", "content": "hello"}
            ]
        }
        result = _estimate_input_tokens(data)
        assert result == 1  # Only "hello" contributes

    def test_estimate_empty_messages_list(self):
        """Test with empty messages list"""
        data = {"messages": []}
        result = _estimate_input_tokens(data)
        assert result == 1

    def test_estimate_string_content(self):
        """Test message with string content (not list)"""
        data = {
            "messages": [
                {"role": "user", "content": "This is a longer message"}  # 27 chars -> 6 tokens
            ]
        }
        result = _estimate_input_tokens(data)
        assert result == 6

    # ==========================================================================
    # _adjust_max_tokens_for_context_window() Tests - Model Detection
    # ==========================================================================

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_adjust_qwen_model_via_router(self, mock_logger):
        """Test Qwen model detected via router lookup"""
        deployment = self.create_mock_deployment("qwen-model", "Qwen/Qwen2-72B")
        mock_router = self.create_mock_router(deployment)

        data = {"model": "qwen-model", "max_tokens": 32000}
        _adjust_max_tokens_for_context_window(data, mock_router)

        # safe_max_tokens_limit = 40960 - 25000 - 500 = 15460
        # 32000 > 15460, so reduce: 32000 * 0.5 = 16000 > 15460, so 16000 * 0.5 = 8000
        assert data["max_tokens"] == 8000

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_adjust_lowercase_qwen(self, mock_logger):
        """Test lowercase qwen model detected"""
        deployment = self.create_mock_deployment("qwen-model", "qwen/qwen2-7b")
        mock_router = self.create_mock_router(deployment)

        data = {"model": "qwen-model", "max_tokens": 32000}
        _adjust_max_tokens_for_context_window(data, mock_router)

        # safe_limit = 15460, 32000 -> 16000 -> 8000
        assert data["max_tokens"] == 8000

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_adjust_gptq_variant(self, mock_logger):
        """Test GPTQ variant detection"""
        deployment = self.create_mock_deployment("gptq-model", "Qwen/GPTQ-Qwen2")
        mock_router = self.create_mock_router(deployment)

        data = {"model": "gptq-model", "max_tokens": 32000}
        _adjust_max_tokens_for_context_window(data, mock_router)

        assert data["max_tokens"] == 8000

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_adjust_int4_variant(self, mock_logger):
        """Test Int4 variant detection"""
        deployment = self.create_mock_deployment("int4-model", "Qwen/Int4-Qwen2")
        mock_router = self.create_mock_router(deployment)

        data = {"model": "int4-model", "max_tokens": 32000}
        _adjust_max_tokens_for_context_window(data, mock_router)

        assert data["max_tokens"] == 8000

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_adjust_non_qwen_model_no_change(self, mock_logger):
        """Test non-Qwen model returns early without modification"""
        deployment = self.create_mock_deployment("gpt4-model", "gpt-4")
        mock_router = self.create_mock_router(deployment)

        data = {"model": "gpt4-model", "max_tokens": 32000}
        original_max_tokens = data["max_tokens"]
        _adjust_max_tokens_for_context_window(data, mock_router)

        assert data["max_tokens"] == original_max_tokens  # No change

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_adjust_router_returns_none_uses_litellm_params(self, mock_logger):
        """Test fallback to litellm_params when router returns None"""
        mock_router = self.create_mock_router(deployment=None)

        data = {
            "model": "qwen-model",
            "max_tokens": 32000,
            "litellm_params": {"model": "Qwen/Qwen2-72B"}
        }
        _adjust_max_tokens_for_context_window(data, mock_router)

        assert data["max_tokens"] == 8000

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_adjust_router_raises_exception_logs_warning(self, mock_logger):
        """Test that router exceptions are caught and logged"""
        mock_router = MagicMock()
        mock_router.get_deployment_by_model_group_name = MagicMock(side_effect=Exception("Router error"))

        data = {
            "model": "qwen-model",
            "max_tokens": 32000,
            "litellm_params": {"model": "Qwen/Qwen2-72B"}
        }
        _adjust_max_tokens_for_context_window(data, mock_router)

        assert data["max_tokens"] == 8000  # Still adjusts via fallback
        mock_logger.warning.assert_called()

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_adjust_no_max_tokens_key(self, mock_logger):
        """Test returns early when max_tokens key is missing"""
        mock_router = self.create_mock_router(
            self.create_mock_deployment("qwen", "Qwen/Qwen2")
        )

        data = {"model": "qwen-model"}
        _adjust_max_tokens_for_context_window(data, mock_router)

        assert "max_tokens" not in data

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_adjust_max_tokens_is_none(self, mock_logger):
        """Test returns early when max_tokens is None"""
        mock_router = self.create_mock_router(
            self.create_mock_deployment("qwen", "Qwen/Qwen2")
        )

        data = {"model": "qwen-model", "max_tokens": None}
        _adjust_max_tokens_for_context_window(data, mock_router)

        assert data["max_tokens"] is None

    # ==========================================================================
    # Token Adjustment Logic Tests - Coefficient-based reduction
    # ==========================================================================

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_adjust_large_max_tokens_reduced_by_coefficient(self, mock_logger):
        """Test large max_tokens is reduced by coefficient until under limit"""
        deployment = self.create_mock_deployment("qwen", "Qwen/Qwen2")
        mock_router = self.create_mock_router(deployment)

        data = {"model": "qwen", "max_tokens": 50000}
        _adjust_max_tokens_for_context_window(data, mock_router)

        # safe_limit = 40960 - 25000 - 500 = 15460
        # 50000 -> 25000 -> 12500 (under limit)
        assert data["max_tokens"] == 12500

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_adjust_under_limit_no_change(self, mock_logger):
        """Test no adjustment when under safe limit"""
        deployment = self.create_mock_deployment("qwen", "Qwen/Qwen2")
        mock_router = self.create_mock_router(deployment)

        data = {"model": "qwen", "max_tokens": 10000}
        _adjust_max_tokens_for_context_window(data, mock_router)

        # safe_limit = 15460, 10000 < 15460, no change
        assert data["max_tokens"] == 10000

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_adjust_just_over_limit(self, mock_logger):
        """Test adjustment when just over safe limit"""
        deployment = self.create_mock_deployment("qwen", "Qwen/Qwen2")
        mock_router = self.create_mock_router(deployment)

        data = {"model": "qwen", "max_tokens": 20000}
        _adjust_max_tokens_for_context_window(data, mock_router)

        # safe_limit = 15460, 20000 -> 10000
        assert data["max_tokens"] == 10000

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_adjust_very_large_request(self, mock_logger):
        """Test adjustment for very large max_tokens request"""
        deployment = self.create_mock_deployment("qwen", "Qwen/Qwen2")
        mock_router = self.create_mock_router(deployment)

        data = {"model": "qwen", "max_tokens": 100000}
        _adjust_max_tokens_for_context_window(data, mock_router)

        # safe_limit = 15460
        # 100000 -> 50000 -> 25000 -> 12500
        assert data["max_tokens"] == 12500

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_adjust_minimum_token_enforcement(self, mock_logger):
        """Test minimum token enforcement (1024 tokens minimum)"""
        deployment = self.create_mock_deployment("qwen", "Qwen/Qwen2")
        mock_router = self.create_mock_router(deployment)

        # Very large max_tokens that would keep reducing
        data = {"model": "qwen", "max_tokens": 200000}
        _adjust_max_tokens_for_context_window(data, mock_router)

        # 200000 -> 100000 -> 50000 -> 25000 -> 12500 (stops here, under limit)
        assert data["max_tokens"] >= 1024

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_adjust_hits_minimum_floor(self, mock_logger):
        """Test that adjustment hits minimum floor of 1024"""
        deployment = self.create_mock_deployment("qwen", "Qwen/Qwen2")
        mock_router = self.create_mock_router(deployment)

        # Set a very small safe limit by using different hidden reserve
        # This simulates a scenario where reduction would go below 1024
        data = {"model": "qwen", "max_tokens": 2000}
        _adjust_max_tokens_for_context_window(data, mock_router)

        # 2000 < safe_limit (15460), so no change
        assert data["max_tokens"] == 2000

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_adjust_exactly_at_limit_boundary(self, mock_logger):
        """Test adjustment at exact limit boundary"""
        deployment = self.create_mock_deployment("qwen", "Qwen/Qwen2")
        mock_router = self.create_mock_router(deployment)

        # safe_limit = 15460
        data = {"model": "qwen", "max_tokens": 15460}
        _adjust_max_tokens_for_context_window(data, mock_router)

        # Exactly at limit, no change
        assert data["max_tokens"] == 15460

    # ==========================================================================
    # Fallback Mechanism Tests
    # ==========================================================================

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_fallback_litellm_params(self, mock_logger):
        """Test fallback to litellm_params when router is None"""
        data = {
            "model": "requested-model",
            "max_tokens": 32000,
            "litellm_params": {"model": "Qwen/Qwen2-72B"}
        }
        _adjust_max_tokens_for_context_window(data, llm_router=None)

        assert data["max_tokens"] == 8000

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_fallback_metadata_deployment(self, mock_logger):
        """Test fallback to metadata.deployment"""
        data = {
            "model": "requested-model",
            "max_tokens": 32000,
            "litellm_params": {},
            "metadata": {"deployment": "qwen/qwen2-7b"}
        }
        _adjust_max_tokens_for_context_window(data, llm_router=None)

        assert data["max_tokens"] == 8000

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_fallback_litellm_metadata_deployment(self, mock_logger):
        """Test fallback to litellm_metadata.deployment"""
        data = {
            "model": "requested-model",
            "max_tokens": 32000,
            "litellm_params": {},
            "metadata": {},
            "litellm_metadata": {"deployment": "Qwen/Qwen2-72B"}
        }
        _adjust_max_tokens_for_context_window(data, llm_router=None)

        assert data["max_tokens"] == 8000

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_fallback_non_qwen_model_no_change(self, mock_logger):
        """Test no adjustment when all fallbacks point to non-Qwen model"""
        data = {
            "model": "requested-model",
            "max_tokens": 32000,
            "litellm_params": {"model": "gpt-4"},
            "metadata": {"deployment": "claude-3"},
            "litellm_metadata": {"deployment": "gemini"}
        }
        original_max_tokens = data["max_tokens"]
        _adjust_max_tokens_for_context_window(data, llm_router=None)

        assert data["max_tokens"] == original_max_tokens

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_fallback_empty_sources(self, mock_logger):
        """Test no adjustment when all sources are empty"""
        data = {
            "model": "requested-model",
            "max_tokens": 32000,
            "litellm_params": {},
            "metadata": {},
            "litellm_metadata": {}
        }
        original_max_tokens = data["max_tokens"]
        _adjust_max_tokens_for_context_window(data, llm_router=None)

        assert data["max_tokens"] == original_max_tokens

    # ==========================================================================
    # Edge Case Tests
    # ==========================================================================

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_dict_litellm_params(self, mock_logger):
        """Test with dict litellm_params (not object)"""
        mock_router = MagicMock()
        deployment = MagicMock()
        deployment.litellm_params = {"model": "Qwen/Qwen2-72B"}
        mock_router.get_deployment_by_model_group_name = MagicMock(return_value=deployment)

        data = {"model": "qwen", "max_tokens": 32000}
        _adjust_max_tokens_for_context_window(data, mock_router)

        assert data["max_tokens"] == 8000

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_object_litellm_params(self, mock_logger):
        """Test with object litellm_params (not dict)"""
        mock_router = MagicMock()
        deployment = MagicMock()
        mock_params = MagicMock()
        mock_params.model = "Qwen/Qwen2-72B"
        deployment.litellm_params = mock_params
        mock_router.get_deployment_by_model_group_name = MagicMock(return_value=deployment)

        data = {"model": "qwen", "max_tokens": 32000}
        _adjust_max_tokens_for_context_window(data, mock_router)

        assert data["max_tokens"] == 8000

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_deployment_without_litellm_params(self, mock_logger):
        """Test deployment without litellm_params attribute"""
        mock_router = MagicMock()
        deployment = MagicMock()
        # Remove litellm_params attribute
        del deployment.litellm_params
        mock_router.get_deployment_by_model_group_name = MagicMock(return_value=deployment)

        data = {
            "model": "qwen",
            "max_tokens": 32000,
            "litellm_params": {"model": "Qwen/Qwen2"}
        }
        _adjust_max_tokens_for_context_window(data, mock_router)

        assert data["max_tokens"] == 8000  # Falls back to data litellm_params

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_empty_model_name(self, mock_logger):
        """Test with empty model name"""
        data = {
            "model": "",
            "max_tokens": 32000,
            "litellm_params": {"model": "Qwen/Qwen2"}
        }
        _adjust_max_tokens_for_context_window(data, llm_router=None)

        assert data["max_tokens"] == 8000

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_max_tokens_zero(self, mock_logger):
        """Test with max_tokens = 0"""
        deployment = self.create_mock_deployment("qwen", "Qwen/Qwen2")
        mock_router = self.create_mock_router(deployment)

        data = {"model": "qwen", "max_tokens": 0}
        _adjust_max_tokens_for_context_window(data, mock_router)

        # 0 < safe_limit (15460), so no change
        assert data["max_tokens"] == 0

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_case_sensitive_qwen_detection(self, mock_logger):
        """Test that Qwen detection is case-insensitive"""
        data = {
            "model": "model",
            "max_tokens": 32000,
            "litellm_params": {"model": "qwen/qwen2-7b"}  # lowercase
        }
        _adjust_max_tokens_for_context_window(data, llm_router=None)

        assert data["max_tokens"] == 8000

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_adjust_single_reduction(self, mock_logger):
        """Test single reduction is enough"""
        deployment = self.create_mock_deployment("qwen", "Qwen/Qwen2")
        mock_router = self.create_mock_router(deployment)

        # 30000 -> 15000 (under limit, single reduction)
        data = {"model": "qwen", "max_tokens": 30000}
        _adjust_max_tokens_for_context_window(data, mock_router)

        # safe_limit = 15460, 30000 * 0.5 = 15000 < 15460
        assert data["max_tokens"] == 15000

    @patch("litellm.proxy.common_request_processing.verbose_proxy_logger")
    def test_adjust_multiple_reductions(self, mock_logger):
        """Test multiple reductions are applied"""
        deployment = self.create_mock_deployment("qwen", "Qwen/Qwen2")
        mock_router = self.create_mock_router(deployment)

        # 100000 -> 50000 -> 25000 -> 12500 (3 reductions)
        data = {"model": "qwen", "max_tokens": 100000}
        _adjust_max_tokens_for_context_window(data, mock_router)

        assert data["max_tokens"] == 12500
