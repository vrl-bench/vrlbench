"""Single-choice admission protects the case budget on every transport path."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrlbench import FactorySpec, ModelSpec, SimulationSpec, WorkloadSpec, simulate
from vrlbench.worker import _ClientProxy


HAS_RUNTIME = importlib.util.find_spec("openai") is not None
INVALID_COUNTS = (True, False, None, 0, -1, 2, 1.0, "1", [], {})


def make_spec(options, **kwargs):
    fixture = ROOT / "examples/memory_study/fixture.py"
    return SimulationSpec(
        WorkloadSpec("teaching_route_control", [{"case_id": "case", "target": "blue"}],
                     factory=FactorySpec(f"{fixture}:create_backend")),
        ModelSpec("scripted", request_options=options,
                  transport=FactorySpec(f"{fixture}:create_no_model_client")),
        trials=6, max_environment_steps=1, **kwargs)


class SingleCompletionValidationTests(unittest.TestCase):
    def test_public_spec_rejects_invalid_counts_in_both_request_layers(self):
        for count in INVALID_COUNTS:
            for options in ({"n": count}, {"extra_body": {"n": count}}):
                with self.subTest(options=options), self.assertRaisesRegex(ValueError, "only one completion"):
                    make_spec(options).validate()
        for options in ({"n": 1, "extra_body": {"n": 2}},
                        {"n": 2, "extra_body": {"n": 1}}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "only one completion"):
                make_spec(options).validate()

    def test_public_spec_keeps_default_and_explicit_single_choice(self):
        for options in ({}, {"n": 1}, {"extra_body": {"n": 1}},
                        {"n": 1, "extra_body": {"n": 1}}):
            with self.subTest(options=options):
                make_spec(options).validate()

    def test_proxy_rejects_mapped_counts_before_creating_client(self):
        def create_client():
            self.fail("invalid completion count must be rejected before creating a client")

        requests = [({"n": count}, {}) for count in INVALID_COUNTS]
        requests += [({}, {"n": count}) for count in INVALID_COUNTS if count is not None]
        requests += [({}, {"extra_body": {"n": count}}) for count in INVALID_COUNTS]
        requests += [({"n": 2}, {"extra_body": {"n": 1}}),
                     ({"n": 1}, {"extra_body": {"n": 2}}),
                     ({"extra_body": {"n": 2}}, {"n": 1})]
        for kwargs, options in requests:
            with self.subTest(kwargs=kwargs, options=options), self.assertRaisesRegex(ValueError, "only one completion"):
                _ClientProxy(create_client, options).chat.completions.create(**kwargs)


@unittest.skipUnless(HAS_RUNTIME, "Install the runtime extra for transport tests")
class SingleCompletionTransportTests(unittest.TestCase):
    def test_proxy_checks_final_extra_body_after_mapping(self):
        captured = []
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: captured.append(kwargs))))
        proxy = _ClientProxy(lambda: client, {"extra_body": {"n": 1, "mapped": True}})
        proxy.chat.completions.create(extra_body={"n": 2, "original": True})
        self.assertEqual(captured, [{"extra_body": {"n": 1, "original": True, "mapped": True}}])

    def test_worker_configuration_rejects_before_loading_transport(self):
        from vrlbench.worker import configure_model

        for options in ({"n": 2}, {"extra_body": {"n": 2}},
                        {"n": 2, "extra_body": {"n": 1}}):
            with self.subTest(options=options), patch("vrlbench.plugins.load_factory") as loader:
                with self.assertRaisesRegex(ValueError, "only one completion"):
                    configure_model({"request_options": options,
                                     "transport": {"entrypoint": "unused:create"}})
                loader.assert_not_called()

    def test_simulate_single_choice_stays_within_case_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plugin = root / "single_choice.py"
            plugin.write_text(textwrap.dedent('''
                from dataclasses import asdict
                import json
                from pathlib import Path
                from types import SimpleNamespace as NS
                class Completion:
                    def __init__(self, path): self.path = path
                    def create(self, **kwargs):
                        with Path(self.path).open('a') as output:
                            output.write(json.dumps(kwargs) + '\\n')
                        n = kwargs.get('extra_body', {}).get('n', kwargs.get('n', 1))
                        completion = n * kwargs['max_tokens']
                        return NS(usage=NS(prompt_tokens=10, completion_tokens=completion,
                                           total_tokens=10 + completion),
                                  choices=[NS(message=NS(content='SUCCESS')) for _ in range(n)])
                def create_client(path):
                    return NS(chat=NS(completions=Completion(path)))
                class Backend:
                    def run(self, spec, runtime):
                        from vrlbench.runtime import llm
                        llm._estimate_message_tokens = lambda messages, model: 10
                        episode = runtime.run_episode(
                            lambda: llm.complete('offline prompt', model=runtime.model),
                            lambda trace: trace == 'SUCCESS')
                        return [{'case_id': 'case', 'task': 'single_choice',
                                 'trials': [asdict(trial) for trial in episode.trials],
                                 'resource_usage': episode.resource_usage}]
                def create_backend(): return Backend()
            '''))
            for index, options in enumerate(({}, {"n": 1}, {"extra_body": {"n": 1}},
                                             {"n": 1, "extra_body": {"n": 1}})):
                calls = root / f"calls-{index}.jsonl"
                spec = SimulationSpec(
                    WorkloadSpec("single_choice", ["case"], factory=FactorySpec(f"{plugin}:create_backend")),
                    ModelSpec("scripted", request_options=options,
                              transport=FactorySpec(f"{plugin}:create_client", {"path": str(calls)})),
                    trials=2, max_environment_steps=15, max_total_tokens=100, output_dir=str(root / f"run-{index}"),
                )
                result = simulate(spec).require_complete()
                self.assertEqual(result.resources["api_calls"], 1)
                self.assertEqual(result.resources["total_tokens"], 100)
                self.assertTrue(result.resources["provider_usage_complete"])
                self.assertEqual(len(result.cases[0]["trials"]), 1)
                rows = [json.loads(line) for line in calls.read_text().splitlines()]
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["max_tokens"], 90)

    def test_simulate_rejects_multiple_choices_before_creating_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rejected"
            for options in ({"n": 3}, {"extra_body": {"n": 3}},
                            {"n": 3, "extra_body": {"n": 1}}):
                with self.subTest(options=options), self.assertRaisesRegex(ValueError, "only one completion"):
                    simulate(make_spec(options, output_dir=str(output)))
                self.assertFalse(output.exists())

    def test_actor_call_returns_a_real_empty_response_without_retry(self):
        from vrlbench.runtime import llm
        from vrlbench.runtime.budget import TokenUsageLedger

        for function, options in ((llm.complete, {"retry_on_error": False}),
                                  (llm.generate, {"retry_on_empty": False})):
            for content in ("", None):
                with self.subTest(function=function.__name__, content=content), \
                        patch.object(llm, "get_client") as get_client, \
                        patch.object(llm.time, "sleep") as sleep:
                    create = get_client.return_value.chat.completions.create
                    create.return_value = SimpleNamespace(
                        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=0, total_tokens=7),
                        choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
                    ledger = TokenUsageLedger(require_provider_usage=True)
                    with llm.token_budget_scope(ledger):
                        self.assertEqual(function("prompt", model="scripted", **options), "")
                    self.assertEqual(create.call_count, 1)
                    self.assertEqual(ledger.api_calls, 1)
                    self.assertEqual(ledger.total_tokens, 7)
                    self.assertEqual(ledger.api_failures, 0)
                    sleep.assert_not_called()

    def test_actor_transport_failure_is_not_an_empty_reply(self):
        from vrlbench.runtime import llm
        from vrlbench.runtime.budget import TokenUsageLedger

        for function, options in ((llm.complete, {"retry_on_error": False}),
                                  (llm.generate, {"retry_on_empty": False})):
            for strict in (False, True):
                with self.subTest(function=function.__name__, strict=strict), \
                        patch.object(llm, "get_client") as get_client, \
                        patch.object(llm.time, "sleep") as sleep:
                    failure = ConnectionError("offline provider failure")
                    create = get_client.return_value.chat.completions.create
                    create.side_effect = failure
                    ledger = TokenUsageLedger(require_provider_usage=strict)
                    error_type = llm.ProviderCallUncertain if strict else RuntimeError
                    with llm.token_budget_scope(ledger), self.assertRaises(error_type) as error:
                        function("prompt", model="scripted", **options)
                    self.assertIs(error.exception.__cause__, failure)
                    self.assertEqual(create.call_count, 1)
                    self.assertEqual(ledger.api_failures, 1)
                    self.assertEqual(ledger.api_calls, 0)
                    sleep.assert_not_called()

    def test_default_transport_retry_exhaustion_raises_without_final_sleep(self):
        from vrlbench.runtime import llm

        for function in (llm.complete, llm.generate):
            with self.subTest(function=function.__name__), \
                    patch.object(llm, "get_client") as get_client, \
                    patch.object(llm.time, "sleep") as sleep:
                failure = ConnectionError("offline provider failure")
                create = get_client.return_value.chat.completions.create
                create.side_effect = failure
                with llm.token_budget_scope(None), self.assertRaisesRegex(
                        RuntimeError, "failed after 6 provider attempt") as error:
                    function("prompt", model="scripted")
                self.assertIs(error.exception.__cause__, failure)
                self.assertEqual(create.call_count, 6)
                self.assertEqual([call.args[0] for call in sleep.call_args_list],
                                 [1, 2, 4, 8, 16])

    def test_actor_token_admission_failure_never_reaches_provider(self):
        from vrlbench.runtime import llm
        from vrlbench.runtime.budget import TokenBudgetExceeded, TokenUsageLedger

        for function, options in ((llm.complete, {"retry_on_error": False}),
                                  (llm.generate, {"retry_on_empty": False})):
            with self.subTest(function=function.__name__), \
                    patch.object(llm, "get_client") as get_client, \
                    patch.object(llm, "_estimate_message_tokens", return_value=10):
                ledger = TokenUsageLedger(max_total_tokens=10, require_provider_usage=True)
                with llm.token_budget_scope(ledger), self.assertRaises(TokenBudgetExceeded):
                    function("prompt", model="scripted", **options)
                get_client.return_value.chat.completions.create.assert_not_called()
                self.assertEqual(ledger.api_calls, 0)
                self.assertEqual(ledger.api_failures, 0)

    def test_generate_does_not_mistake_recovered_empty_response_for_transport_failure(self):
        from vrlbench.runtime import llm
        from vrlbench.runtime.budget import TokenUsageLedger

        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=0, total_tokens=7),
            choices=[SimpleNamespace(message=SimpleNamespace(content=""))])
        with patch.object(llm, "get_client") as get_client, \
                patch.object(llm.time, "sleep") as sleep:
            create = get_client.return_value.chat.completions.create
            create.side_effect = [ConnectionError("offline provider failure")] + [response] * 5
            ledger = TokenUsageLedger()
            with llm.token_budget_scope(ledger):
                self.assertEqual(llm.generate("prompt", model="scripted"), "")
            self.assertEqual(create.call_count, 6)
            self.assertEqual(ledger.api_failures, 1)
            self.assertEqual(ledger.api_calls, 5)
            sleep.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
