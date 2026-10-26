"""Model-call costs survive uncommitted cases, exceptions, and worker timeouts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrlbench import FactorySpec, MemorySpec, ModelSpec, SimulationSpec, WorkloadSpec, compare_results, simulate
from vrlbench.provider_usage import JOURNAL_NAME, ProviderUsageJournal, merge_provider_resources, read_provider_usage


class ProviderJournalTests(unittest.TestCase):
    def test_pending_request_keeps_known_subtotal_and_contains_no_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal = ProviderUsageJournal(root)
            request = journal.begin()
            journal.response(request, {"usage": {"prompt_tokens": 7, "completion_tokens": 5,
                                                 "total_tokens": 12},
                                       "choices": [{"message": {"content": "private response"}}],
                                       "headers": {"authorization": "private key"}})
            journal.begin()
            summary = read_provider_usage(root)
            self.assertEqual(summary["api_calls"], 2)
            self.assertIsNone(summary["total_tokens"])
            self.assertEqual(summary["known_subtotals"]["total_tokens"], 12)
            self.assertEqual(summary["pending_requests"], 1)
            self.assertEqual(summary["unknown_usage_calls"], 1)
            self.assertFalse(summary["provider_usage_complete"])
            text = (root / JOURNAL_NAME).read_text()
            self.assertNotIn("private", text)
            self.assertNotIn("choices", text)
            self.assertNotIn("headers", text)

    def test_partial_missing_invalid_and_truncated_usage_never_becomes_zero_cost(self):
        for usage in ({"prompt_tokens": 7}, None,
                      {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 2}):
            with self.subTest(usage=usage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                journal = ProviderUsageJournal(root)
                journal.response(journal.begin(), {"usage": usage})
                summary = read_provider_usage(root)
                self.assertIsNone(summary["total_tokens"])
                self.assertFalse(summary["provider_usage_complete"])
                self.assertEqual(summary["unknown_usage_calls"], 1)
                if usage == {"prompt_tokens": 7}:
                    self.assertEqual(summary["known_subtotals"]["prompt_tokens"], 7)
                with (root / JOURNAL_NAME).open("a") as handle:
                    handle.write('{"event":')
                self.assertFalse(read_provider_usage(root)["journal_valid"])
                self.assertIsNone(read_provider_usage(root)["api_calls"])

    def test_shared_first_trial_is_not_a_new_request_and_custom_cost_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ProviderUsageJournal(root)
            journal = read_provider_usage(root)
            cases = {"api_calls": 9, "total_tokens": 999,
                     "known_subtotals": {"total_tokens": 999, "environment_steps": 3},
                     "known_cases": {"total_tokens": 1, "environment_steps": 1}}
            native = merge_provider_resources(cases, journal, include_zero=True)
            self.assertEqual(native["api_calls"], 0)
            self.assertEqual(native["total_tokens"], 0)
            self.assertEqual(native["known_subtotals"]["environment_steps"], 3)
            self.assertEqual(cases["total_tokens"], 999)
            custom = merge_provider_resources(cases, journal, include_zero=False)
            self.assertEqual(custom["total_tokens"], 999)


@unittest.skipUnless(importlib.util.find_spec("openai"), "Install the runtime extra for worker tests")
class ProviderJournalWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.plugin = self.root / "fixture.py"
        self.plugin.write_text(textwrap.dedent('''
            import time
            from types import SimpleNamespace
            class Completion:
                def __init__(self, mode): self.mode, self.calls = mode, 0
                def create(self, **kwargs):
                    self.calls += 1
                    if self.calls == 2 and self.mode == 'transport_failure':
                        raise RuntimeError('private provider error body')
                    if self.calls == 2 and self.mode == 'timeout':
                        time.sleep(30)
                    return SimpleNamespace(model='scripted',
                        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=5, total_tokens=12),
                        choices=[SimpleNamespace(message=SimpleNamespace(content='private response'))])
            def create_client(mode):
                return SimpleNamespace(chat=SimpleNamespace(completions=Completion(mode)))
            class Backend:
                def __init__(self, mode): self.mode = mode
                def run(self, spec, runtime):
                    from vrlbench.runtime import llm
                    from vrlbench.runtime.budget import TokenBudgetExceeded, TokenUsageLedger
                    if self.mode == 'journal_failure':
                        def fail_write(*args): raise OSError('fixture disk write failure')
                        llm.get_client().journal.response = fail_write
                    with llm.token_budget_scope(TokenUsageLedger(require_provider_usage=True)):
                        llm.complete('private prompt', model='scripted')
                        if self.mode != 'complete':
                            llm.complete('second private prompt', model='scripted')
                            raise TokenBudgetExceeded('stop after real calls before committing a case')
                    # A copied first-trial ledger is intentionally larger than
                    # this run's actual requests. Do not add the two together.
                    return [{'case_id': 'a', 'task': 'fixture',
                             'trials': [{'trial': 0, 'success': True}],
                             'resource_usage': {'api_calls': 99, 'total_tokens': 999,
                                                'provider_usage_complete': True}}]
            def create_backend(mode): return Backend(mode)
        '''))

    def run_fixture(self, mode):
        return simulate(SimulationSpec(
            WorkloadSpec("fixture", ["a"], factory=FactorySpec(
                f"{self.plugin}:create_backend", {"mode": mode})),
            ModelSpec("scripted", transport=FactorySpec(
                f"{self.plugin}:create_client", {"mode": mode})),
            MemorySpec("retry"), trials=2, output_dir=str(self.root / mode),
            timeout_seconds=4 if mode == "timeout" else 20,
        ))

    def test_failed_uncommitted_case_retains_all_reported_model_usage(self):
        result = self.run_fixture("budget_failure")
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.counts["completed"], 0)
        self.assertEqual(result.cases, [])
        self.assertIsNone(result.metrics["sr_at_t"])
        self.assertEqual(result.resources["api_calls"], 2)
        self.assertEqual(result.resources["total_tokens"], 24)
        self.assertTrue(result.resources["provider_usage_complete"])

    def test_transport_failure_does_not_retry_or_invent_unknown_tokens(self):
        result = self.run_fixture("transport_failure")
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.counts["completed"], 0)
        self.assertEqual(result.resources["api_calls"], 2)
        self.assertIsNone(result.resources["total_tokens"])
        self.assertEqual(result.resources["known_subtotals"]["total_tokens"], 12)
        requests = result.resources["provider_requests"]
        self.assertEqual(requests["failed_requests"], 1)
        self.assertEqual(requests["unknown_usage_calls"], 1)
        self.assertNotIn("private", (result.output_dir / JOURNAL_NAME).read_text())

    def test_journal_write_failure_stops_without_another_model_call(self):
        result = self.run_fixture("journal_failure")
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.to_dict()["error"]["type"], "ProviderUsagePersistenceError")
        self.assertEqual(result.resources["api_calls"], 1)
        self.assertIsNone(result.resources["total_tokens"])
        self.assertEqual(result.resources["provider_requests"]["pending_requests"], 1)

    def test_timeout_keeps_completed_request_and_marks_inflight_request_unknown(self):
        result = self.run_fixture("timeout")
        self.assertEqual(result.status, "timeout")
        self.assertEqual(result.counts["completed"], 0)
        self.assertIsNone(result.metrics["sr_at_t"])
        self.assertEqual(result.resources["api_calls"], 2)
        self.assertIsNone(result.resources["total_tokens"])
        self.assertEqual(result.resources["known_subtotals"]["total_tokens"], 12)
        self.assertEqual(result.resources["provider_requests"]["pending_requests"], 1)

    def test_complete_case_is_not_double_counted_and_comparison_keeps_actual_cost(self):
        result = self.run_fixture("complete").require_complete()
        self.assertEqual(result.cases[0]["resource_usage"]["total_tokens"], 999)
        self.assertEqual(result.resources["total_tokens"], 12)
        self.assertEqual(result.resources["api_calls"], 1)
        self.assertEqual(compare_results([result])["rows"][0]["resources"], result.resources)
        loaded = json.loads((result.output_dir / "evaluation.json").read_text())
        self.assertEqual(loaded["resources"], result.resources)


if __name__ == "__main__":
    unittest.main()
