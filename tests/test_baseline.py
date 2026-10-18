from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
import unittest

from tests.run_baseline import known_sum, quality_signals, trace_metrics


class BaselineMetricsTests(unittest.TestCase):
    def span(self, kind, tokens, cost):
        started = datetime.now(timezone.utc)
        return SimpleNamespace(
            run_type=kind, total_tokens=tokens, prompt_tokens=tokens - 5,
            completion_tokens=5, total_cost=cost,
            start_time=started, end_time=started + timedelta(seconds=2),
        )

    def test_sums_llm_leaves_without_counting_parent_aggregates_or_tools(self):
        root = self.span("chain", 9000, Decimal("99"))
        spans = [root, self.span("llm", 15, Decimal("0.01")),
                 self.span("llm", 25, Decimal("0.02")), self.span("tool", 9000, Decimal("99"))]
        result = trace_metrics(root, spans)
        self.assertEqual(result["llm_calls"], 2)
        self.assertEqual(result["github_tool_calls"], 1)
        self.assertEqual(result["total_tokens"], 40)
        self.assertEqual(result["cost_usd"], "0.03")
        self.assertEqual(result["root_reported_tokens"], 9000)

    def test_missing_leaf_cost_stays_unknown_even_if_parent_has_cost(self):
        root = self.span("chain", 9000, Decimal("99"))
        result = trace_metrics(root, [root, self.span("llm", 15, None)])
        self.assertIsNone(result["cost_usd"])
        self.assertEqual(result["llm_usage_coverage"]["with_cost"], 0)
        self.assertEqual(known_sum([0, 0], money=True), "0")
        self.assertIsNone(known_sum([Decimal("0.1"), None], money=True))

    def test_missing_tokens_do_not_produce_partial_totals(self):
        root = self.span("chain", 9000, None)
        missing = self.span("llm", 15, None)
        missing.total_tokens = None
        result = trace_metrics(root, [self.span("llm", 25, None), missing])
        self.assertIsNone(result["total_tokens"])
        self.assertEqual(result["llm_usage_coverage"]["with_total_tokens"], 1)

    def test_preserves_quality_contradictions_without_turning_critic_into_ground_truth(self):
        result = quality_signals({
            "ranking": {"best_repository": "owner/a", "top_repositories": [
                {"full_name": "owner/b"}, {"full_name": "owner/b"}]},
            "critic_result": {"passed": True, "issues": []},
        })
        self.assertFalse(result["best_in_top"])
        self.assertFalse(result["unique_top_repositories"])
        self.assertEqual(result["independent_quality_review"], "pending")


if __name__ == "__main__":
    unittest.main()
