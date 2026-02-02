"""
Test max_tokens adjustment based on model configuration.

This test verifies that when a model has max_tokens_limit configured,
requests with max_tokens exceeding that limit are automatically reduced.
"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
import pytest

from litellm.proxy.common_request_processing import _adjust_max_tokens_for_context_window


class TestMaxTokensAdjustment:
    """Test max_tokens adjustment functionality"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup temporary log directory and cleanup after each test"""
        # Create temporary directory for logs
        self.temp_dir = tempfile.mkdtemp()
        self.logs_dir = Path(self.temp_dir) / "logs"
        self.logs_dir.mkdir()

        # Set environment variable for log directory
        original_log_dir = os.environ.get("LITELLM_LOG_DIR")
        os.environ["LITELLM_LOG_DIR"] = str(self.logs_dir)

        yield

        # Cleanup: remove temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

        # Restore original env var
        if original_log_dir is not None:
            os.environ["LITELLM_LOG_DIR"] = original_log_dir
        elif "LITELLM_LOG_DIR" in os.environ:
            del os.environ["LITELLM_LOG_DIR"]

    def _create_mock_proxy_config(self, model_list):
        """Create a mock ProxyConfig with given model list"""
        mock_config = MagicMock()
        mock_config.config = {
            "model_list": model_list,
            "litellm_settings": {}
        }
        return mock_config

    def _create_mock_router(self, model_list):
        """Create a mock Router with given model list"""
        mock_router = MagicMock()
        mock_router.get_model_list = MagicMock(return_value=model_list)
        return mock_router

    def test_max_tokens_adjusted_when_limit_configured(self):
        """Test that max_tokens is reduced when it exceeds configured limit"""
        model_list = [
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "custom_openai/test-model"
                },
                "max_tokens_limit": 10000
            }
        ]

        proxy_config = self._create_mock_proxy_config(model_list)
        llm_router = self._create_mock_router(model_list)

        data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 50000  # Exceeds limit
        }

        _adjust_max_tokens_for_context_window(data, llm_router, proxy_config)

        # max_tokens should be reduced
        assert data["max_tokens"] < 50000
        assert data["max_tokens"] >= 1024  # MIN_MAX_TOKENS

    def test_max_tokens_not_adjusted_when_under_limit(self):
        """Test that max_tokens is NOT adjusted when it's under the limit"""
        model_list = [
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "custom_openai/test-model"
                },
                "max_tokens_limit": 50000
            }
        ]

        proxy_config = self._create_mock_proxy_config(model_list)
        llm_router = self._create_mock_router(model_list)

        original_max_tokens = 5000
        data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": original_max_tokens
        }

        _adjust_max_tokens_for_context_window(data, llm_router, proxy_config)

        # max_tokens should remain unchanged
        assert data["max_tokens"] == original_max_tokens

    def test_max_tokens_not_adjusted_when_no_limit_configured(self):
        """Test that max_tokens is NOT adjusted when model has no limit configured"""
        model_list = [
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "custom_openai/test-model"
                }
            }
        ]

        proxy_config = self._create_mock_proxy_config(model_list)
        llm_router = self._create_mock_router(model_list)

        original_max_tokens = 50000
        data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": original_max_tokens
        }

        _adjust_max_tokens_for_context_window(data, llm_router, proxy_config)

        # max_tokens should remain unchanged
        assert data["max_tokens"] == original_max_tokens

    def test_max_tokens_reduction_algorithm(self):
        """Test the coefficient-based reduction algorithm"""
        model_list = [
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "custom_openai/test-model"
                },
                "max_tokens_limit": 16000
            }
        ]

        proxy_config = self._create_mock_proxy_config(model_list)
        llm_router = self._create_mock_router(model_list)

        # With limit 16000 and safety margin 500, safe_limit = 15500
        # Requesting 50000 tokens should be reduced by multiplying 0.5 repeatedly
        # 50000 -> 25000 -> 12500 (under 15500)
        data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 50000
        }

        _adjust_max_tokens_for_context_window(data, llm_router, proxy_config)

        # Should be reduced to 12500 (50000 * 0.5 * 0.5)
        assert data["max_tokens"] == 12500

    def test_max_tokens_clamped_to_minimum(self):
        """Test that max_tokens is clamped to MIN_MAX_TOKENS (1024) when needed"""
        model_list = [
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "custom_openai/test-model"
                },
                "max_tokens_limit": 1100  # Very small limit, safe_limit = 600
            }
        ]

        proxy_config = self._create_mock_proxy_config(model_list)
        llm_router = self._create_mock_router(model_list)

        # With limit 1100 and safety margin 500, safe_limit = 600
        # Requesting 50000: 50000 -> 25000 -> 12500 -> 6250 -> 3125 -> 1562 -> 781
        # 781 < 600? No. 781 * 0.5 = 390 < 1024, so clamp to 1024
        data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 50000
        }

        _adjust_max_tokens_for_context_window(data, llm_router, proxy_config)

        # Should be clamped to minimum
        assert data["max_tokens"] == 1024

    def test_no_adjustment_when_max_tokens_not_specified(self):
        """Test that nothing happens when max_tokens is not in the request"""
        model_list = [
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "custom_openai/test-model"
                },
                "max_tokens_limit": 10000
            }
        ]

        proxy_config = self._create_mock_proxy_config(model_list)
        llm_router = self._create_mock_router(model_list)

        data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}]
            # No max_tokens
        }

        _adjust_max_tokens_for_context_window(data, llm_router, proxy_config)

        # max_tokens should not be added
        assert "max_tokens" not in data

    def test_model_matching_by_actual_model_name(self):
        """Test that model matching works when using the actual model name in litellm_params"""
        model_list = [
            {
                "model_name": "alias-name",
                "litellm_params": {
                    "model": "custom_openai/actual-model-name"
                },
                "max_tokens_limit": 10000
            }
        ]

        proxy_config = self._create_mock_proxy_config(model_list)
        llm_router = self._create_mock_router(model_list)

        # Request using the actual model name from litellm_params
        data = {
            "model": "custom_openai/actual-model-name",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 50000
        }

        _adjust_max_tokens_for_context_window(data, llm_router, proxy_config)

        # Should still find the model and adjust
        assert data["max_tokens"] < 50000

    def test_fallback_to_proxy_config_when_router_unavailable(self):
        """Test that proxy_config is used as fallback when llm_router is None"""
        model_list = [
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "custom_openai/test-model"
                },
                "max_tokens_limit": 10000
            }
        ]

        proxy_config = self._create_mock_proxy_config(model_list)
        # No llm_router

        data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 50000
        }

        _adjust_max_tokens_for_context_window(data, None, proxy_config)

        # Should still work using proxy_config
        assert data["max_tokens"] < 50000

    def test_multiple_models_with_different_limits(self):
        """Test that different models have independent limits"""
        model_list = [
            {
                "model_name": "model-a",
                "litellm_params": {
                    "model": "custom_openai/model-a"
                },
                "max_tokens_limit": 5000
            },
            {
                "model_name": "model-b",
                "litellm_params": {
                    "model": "custom_openai/model-b"
                },
                "max_tokens_limit": 50000
            },
            {
                "model_name": "model-c",
                "litellm_params": {
                    "model": "custom_openai/model-c"
                }
            }
        ]

        proxy_config = self._create_mock_proxy_config(model_list)
        llm_router = self._create_mock_router(model_list)

        # Test model-a with small limit
        data_a = {
            "model": "model-a",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 50000
        }
        _adjust_max_tokens_for_context_window(data_a, llm_router, proxy_config)
        assert data_a["max_tokens"] < 50000

        # Test model-b with large limit
        data_b = {
            "model": "model-b",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 50000
        }
        original_b = data_b["max_tokens"]
        _adjust_max_tokens_for_context_window(data_b, llm_router, proxy_config)
        # Should not be reduced (50000 < 50000 - 500 = 49500 is false, but close)
        # Actually 50000 > 49500, so it WILL be reduced
        assert data_b["max_tokens"] < original_b

        # Test model-c with no limit
        data_c = {
            "model": "model-c",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 50000
        }
        original_c = data_c["max_tokens"]
        _adjust_max_tokens_for_context_window(data_c, llm_router, proxy_config)
        assert data_c["max_tokens"] == original_c  # No change


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
