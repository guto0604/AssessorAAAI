import os
import unittest
from unittest.mock import patch

from core import tavily_client


class TavilyClientTests(unittest.TestCase):
    def test_env_key_has_precedence_over_session_key(self):
        with patch.dict(os.environ, {"TAVILY_API_KEY": "env-key"}, clear=False):
            with patch.object(tavily_client.st, "session_state", {tavily_client.SESSION_TAVILY_KEY: "session-key"}):
                self.assertEqual(tavily_client.get_effective_tavily_api_key(), "env-key")

    @patch("core.tavily_client.requests.post")
    def test_search_uses_authorization_header(self, post_mock):
        post_mock.return_value.json.return_value = {"results": []}
        post_mock.return_value.raise_for_status.return_value = None

        with patch.object(tavily_client.st, "session_state", {tavily_client.SESSION_TAVILY_KEY: "session-key"}):
            tavily_client.search_tavily("mercado")

        _, kwargs = post_mock.call_args
        self.assertEqual(kwargs["headers"], {"Authorization": "Bearer session-key"})


if __name__ == "__main__":
    unittest.main()
