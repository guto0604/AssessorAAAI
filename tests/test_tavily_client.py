import os
import unittest
from unittest.mock import patch

from core import tavily_client


class TavilyClientTests(unittest.TestCase):
    def test_env_key_has_precedence_over_session_key(self):
        with patch.dict(os.environ, {"TAVILY_API_KEY": "env-key"}, clear=False):
            with patch.object(tavily_client.st, "session_state", {tavily_client.SESSION_TAVILY_KEY: "session-key"}):
                self.assertEqual(tavily_client.get_effective_tavily_api_key(), "env-key")

    @patch("core.tavily_client.TavilyClient")
    def test_search_uses_tavily_client_with_api_key(self, client_mock):
        client_mock.return_value.search.return_value = {"results": []}

        with patch.object(tavily_client.st, "session_state", {tavily_client.SESSION_TAVILY_KEY: "session-key"}):
            tavily_client.search_tavily("mercado")

        client_mock.assert_called_once_with(api_key="session-key")
        _, kwargs = client_mock.return_value.search.call_args
        self.assertEqual(kwargs["query"], "mercado")


if __name__ == "__main__":
    unittest.main()
