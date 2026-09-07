import unittest

from fastapi.testclient import TestClient

import reposcouts.api.app as app_module
from reposcouts.api.run_manager import RunManager

from tests.test_run_manager import SuccessfulGraph


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app_module.app)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_rejects_too_short_request(self):
        response = self.client.post(
            "/api/runs",
            json={"user_request": "    "},
        )

        self.assertEqual(response.status_code, 422)

    def test_create_and_stream_a_complete_run(self):
        original_manager = app_module.run_manager
        app_module.run_manager = RunManager(graph=SuccessfulGraph())

        try:
            create_response = self.client.post(
                "/api/runs",
                json={"user_request": "Find a beginner LangGraph project"},
            )
            self.assertEqual(create_response.status_code, 202)
            run_id = create_response.json()["run_id"]

            with self.client.stream(
                "GET",
                f"/api/runs/{run_id}/events",
            ) as stream_response:
                self.assertEqual(stream_response.status_code, 200)
                event_lines = [
                    line
                    for line in stream_response.iter_lines()
                    if line.startswith("event: ")
                ]

            self.assertEqual(event_lines[0], "event: run.started")
            self.assertIn("event: node.completed", event_lines)
            self.assertEqual(event_lines[-1], "event: run.completed")
        finally:
            app_module.run_manager = original_manager


if __name__ == "__main__":
    unittest.main()
