import os
import unittest
from unittest.mock import patch

from core import tavily_client


class TavilyClientTests(unittest.TestCase):
    def test_env_key_has_precedence_over_session_key(self):
        """Test env key has precedence over session key.

        Returns:
            Valor de retorno da função.
        """
        with patch.dict(os.environ, {"TAVILY_API_KEY": "env-key"}, clear=False):
            with patch.object(tavily_client.st, "session_state", {tavily_client.SESSION_TAVILY_KEY: "session-key"}):
                self.assertEqual(tavily_client.get_effective_tavily_api_key(), "env-key")

    @patch("core.tavily_client.TavilyClient")
    def test_search_uses_tavily_client_with_api_key(self, client_mock):
        """Test search uses tavily client with api key.

        Args:
            client_mock: Descrição do parâmetro `client_mock`.

        Returns:
            Valor de retorno da função.
        """
        client_mock.return_value.search.return_value = {"results": []}

        with patch.object(tavily_client.st, "session_state", {tavily_client.SESSION_TAVILY_KEY: "session-key"}):
            tavily_client.search_tavily("mercado")

        client_mock.assert_called_once_with(api_key="session-key")
        _, kwargs = client_mock.return_value.search.call_args
        self.assertEqual(kwargs["query"], "mercado")
        self.assertEqual(kwargs["days"], 7)
        self.assertNotIn("start_date", kwargs)

    @patch("core.tavily_client.TavilyClient")
    def test_search_uses_lightweight_payload_when_requested(self, client_mock):
        """Test search uses lightweight payload when requested.

        Args:
            client_mock: Descrição do parâmetro `client_mock`.

        Returns:
            Valor de retorno da função.
        """
        client_mock.return_value.search.return_value = {"results": []}

        with patch.object(tavily_client.st, "session_state", {tavily_client.SESSION_TAVILY_KEY: "session-key"}):
            tavily_client.search_tavily("mercado", lightweight=True)

        _, kwargs = client_mock.return_value.search.call_args
        self.assertFalse(kwargs["include_raw_content"])

    @patch("core.tavily_client.TavilyClient")
    def test_search_retries_with_smaller_payload_on_timeout(self, client_mock):
        """Test search retries with smaller payload on timeout.

        Args:
            client_mock: Descrição do parâmetro `client_mock`.

        Returns:
            Valor de retorno da função.
        """
        client_mock.return_value.search.side_effect = [Exception("Read timed out"), {"results": []}]

        with patch.object(tavily_client.st, "session_state", {tavily_client.SESSION_TAVILY_KEY: "session-key"}):
            tavily_client.search_tavily("mercado", num_results=12)

        self.assertEqual(client_mock.return_value.search.call_count, 2)
        _, second_call_kwargs = client_mock.return_value.search.call_args_list[1]
        self.assertEqual(second_call_kwargs["search_depth"], "basic")
        self.assertFalse(second_call_kwargs["include_raw_content"])
        self.assertEqual(second_call_kwargs["max_results"], 6)


if __name__ == "__main__":
    unittest.main()
