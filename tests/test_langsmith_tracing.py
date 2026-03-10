import unittest
from unittest.mock import patch

from langsmith_tracing import LangSmithTracer


class LangSmithTracerTests(unittest.TestCase):
    @patch.object(LangSmithTracer, "_send_json")
    def test_start_run_payload(self, mock_send_json):
        tracer = LangSmithTracer(api_key="k", enabled=True)
        run_id = tracer.start_run(name="r", run_type="chain", inputs={"a": 1})
        self.assertTrue(run_id)
        _method, _url, payload = mock_send_json.call_args.args
        self.assertEqual(payload["session_name"], tracer.project_name)
        self.assertEqual(payload["trace_id"], payload["id"])
        self.assertNotIn("dotted_order", payload)

    @patch.object(LangSmithTracer, "_send_json")
    def test_end_run_sends_status(self, mock_send_json):
        tracer = LangSmithTracer(api_key="k", enabled=True)
        tracer.end_run("run-1", status="completed", outputs={"ok": True})
        _method, _url, payload = mock_send_json.call_args.args
        self.assertEqual(payload["status"], "completed")


if __name__ == "__main__":
    unittest.main()
