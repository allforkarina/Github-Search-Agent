from copy import deepcopy
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from reposcouts.observability.metrics import build_run_summary
from reposcouts.observability.tracing import (
    sanitize_trace_payload,
    build_run_tracer,
)


class RunSummaryTests(unittest.TestCase):
    def test_classifies_execution_and_critic_outcomes(self):
        cases = [
            ("failed", 0, 2, True, "failed"),
            ("cancelled", 1, 2, True, "cancelled"),
            ("completed", 0, 2, None, "unreviewed"),
            ("completed", 2, 2, True, "passed"),
            ("completed", 2, 2, False, "retry_exhausted"),
            ("completed", 3, 2, False, "retry_exhausted"),
            ("completed", 1, 2, False, "critic_rejected"),
        ]
        for status, retries, limit, passed, outcome in cases:
            with self.subTest(status=status, retries=retries, passed=passed):
                self.assertEqual(
                    build_run_summary(
                        execution_status=status,
                        retry_count=retries,
                        max_retries=limit,
                        critic_passed=passed,
                    ),
                    {
                        "execution_status": status,
                        "retry_count": retries,
                        "critic_passed": passed,
                        "outcome": outcome,
                    },
                )

    def test_rejects_non_terminal_statuses(self):
        for status in ("running", "cancelling", ""):
            with self.subTest(status=status):
                with self.assertRaises(ValueError):
                    build_run_summary(
                        execution_status=status,
                        retry_count=0,
                        max_retries=2,
                        critic_passed=None,
                    )


class TraceSummaryTests(unittest.TestCase):
    def test_merges_summary_without_replacing_results_or_version_metadata(self):
        from langsmith import Client, tracing_context

        metadata = {"prompt_version": "v1", "retry_count": 0}
        outputs = {"ranking": {"best_repository": "example/agent"}}
        original_outputs = deepcopy(outputs)
        summary = {"retry_count": 2, "outcome": "retry_exhausted"}
        provider = Mock(return_value=summary)
        client = Client(
            api_key="offline-key", api_url="http://127.0.0.1:1",
            auto_batch_tracing=False,
        )
        self.addCleanup(client.close, timeout=0)
        tracer = build_run_tracer(
            client=client, project_name="offline-test", summary_provider=provider,
        )
        # Framework copies must retain this request's summary provider.
        tracer = tracer.copy_with_metadata_defaults(metadata={"ls_agent_type": "test"})
        run_id = uuid4()
        with tracing_context(enabled=True, client=client), \
                patch("requests.Session.request", side_effect=AssertionError("Network forbidden")), \
                patch.object(Client, "create_run"), \
                patch.object(Client, "update_run") as update:
            tracer.on_chain_start(
                {"name": "OfflineRoot"}, {}, run_id=run_id, metadata=metadata,
            )
            tracer.on_chain_end(outputs, run_id=run_id)

        provider.assert_called_once_with("completed")
        update.assert_called_once()
        payload = update.call_args.kwargs
        self.assertEqual(payload["run_id"], run_id)
        self.assertEqual(payload["end_time"], tracer.latest_run.end_time)
        self.assertEqual(payload["outputs"]["ranking"], original_outputs["ranking"])
        self.assertEqual(payload["outputs"]["observability"], summary)
        self.assertEqual(payload["extra"]["metadata"]["prompt_version"], "v1")
        self.assertEqual(payload["extra"]["metadata"]["retry_count"], 2)
        self.assertEqual(outputs, original_outputs)


class SanitizerTests(unittest.TestCase):
    def test_redacts_nested_keys_and_free_text_without_changing_usage(self):
        payload = {
            "Authorization": "Bearer private-value",
            "X-API-Key": "unknown-key",
            "nested": [{"github_token": "unknown-token"}],
            "message": "Bearer opaque.value ghp_example123 known-secret",
            "usage_metadata": {"input_tokens": 10, "output_tokens": 5},
        }
        original = deepcopy(payload)
        cleaned = sanitize_trace_payload(payload, secrets=("known-secret",))
        self.assertEqual(cleaned["Authorization"], "[REDACTED]")
        self.assertEqual(cleaned["X-API-Key"], "[REDACTED]")
        self.assertEqual(cleaned["nested"][0]["github_token"], "[REDACTED]")
        self.assertEqual(cleaned["message"], "Bearer [REDACTED] [REDACTED] [REDACTED]")
        self.assertEqual(cleaned["usage_metadata"], payload["usage_metadata"])
        self.assertEqual(payload, original)

    def test_redacts_before_truncating_and_handles_overlapping_secrets(self):
        cleaned = sanitize_trace_payload(
            {"text": "prefix key-long-secret trailing"},
            secrets=("key", "key-long-secret"),
            max_string_chars=18,
        )
        self.assertEqual(cleaned["text"], "prefix [REDACTED] [TRUNCATED]")

    def test_installed_sdk_scrubs_inputs_outputs_metadata_and_errors(self):
        # Exercise the pinned SDK's anonymizer hooks without opening a connection.
        from langsmith import Client

        client = Client(
            api_key="lsv2_pt_offline_test",
            api_url="http://127.0.0.1:1",
            auto_batch_tracing=False,
            anonymizer=lambda payload: sanitize_trace_payload(
                payload, secrets=("private-secret",)
            ),
        )
        self.addCleanup(client.close, timeout=0)
        payload = {"message": "private-secret", "input_tokens": 12}
        for method in (
            client._hide_run_inputs, client._hide_run_outputs,
            client._hide_run_metadata,
        ):
            with self.subTest(method=method.__name__):
                self.assertEqual(method(payload), {
                    "message": "[REDACTED]", "input_tokens": 12,
                })
        self.assertEqual(
            client._hide_run_error("Authorization: Bearer private-secret"),
            "Authorization: Bearer [REDACTED]",
        )


if __name__ == "__main__":
    unittest.main()
