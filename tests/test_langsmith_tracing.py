import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


if "requests" not in sys.modules:
    def _ok_response(*args, **kwargs):
        return SimpleNamespace(raise_for_status=lambda: None)

    sys.modules["requests"] = SimpleNamespace(post=_ok_response, patch=_ok_response)

from langsmith_tracing import LangSmithTracer


class LangSmithTracerConfigTests(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "LANGCHAIN_API_KEY": "key-from-langchain",
            "LANGCHAIN_ENDPOINT": "https://custom.endpoint",
            "LANGCHAIN_PROJECT": "project-from-langchain",
            "LANGCHAIN_TRACING_V2": "true",
        },
        clear=True,
    )
    def test_uses_langchain_aliases(self):
        tracer = LangSmithTracer(api_key="", enabled=True)
        self.assertTrue(tracer.enabled)
        self.assertEqual(tracer.api_key, "key-from-langchain")
        self.assertEqual(tracer.base_url, "https://custom.endpoint")
        self.assertEqual(tracer.project_name, "project-from-langchain")

    @patch.dict(
        os.environ,
        {
            "LANGSMITH_API_KEY": "key",
            "LANGCHAIN_TRACING_V2": "false",
            "LANGSMITH_TRACING": "false",
        },
        clear=True,
    )
    def test_disables_tracing_when_flags_false(self):
        tracer = LangSmithTracer(api_key="", enabled=True)
        self.assertFalse(tracer.enabled)

    @patch.dict(
        os.environ,
        {
            "LANGSMITH_API_KEY": "key",
            "LANGCHAIN_TRACING_V2": "true",
        },
        clear=True,
    )
    @patch("langsmith_tracing.requests.post")
    def test_start_run_with_langchain_tracing_v2(self, mock_post):
        mock_post.return_value.raise_for_status.return_value = None
        tracer = LangSmithTracer(api_key="", enabled=True)

        run_id = tracer.start_run(
            name="healthcheck",
            run_type="tool",
            inputs={"source": "test"},
        )

        self.assertIsNotNone(run_id)
        self.assertTrue(mock_post.called)


if __name__ == "__main__":
    unittest.main()
