"""Synthetic API walkthrough: use a reference method in an external agent loop."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from types import SimpleNamespace
from typing import Any

from vrlbench.methods import (
    ActorContext, CaseConfig, TrialRecord, load_method, render_trajectory,
)


TASK = "Buy a grey sofa with the requested arm style."
INITIAL_OBSERVATION = "WebShop search page."
ACTOR_SYSTEM = "Complete the shopping task using the available public tools."
REFLECTION = "Check the product's arm style and select the requested colour before buying."
JOINT_RESPONSE = """
Task-level policy choice
  Decision: Does the listing match both requested attributes?
    Policy A: Inspect the arm style, then select grey before buying.
      Grounding: Trial 1 bought the default option without checking its attributes.
      NEXT: 1 complete trial.
    Policy B: Search for an alternative listing if the arm style does not match.
      Grounding: The failed trace does not establish that all listings are unsuitable.
      RESERVE: 1 complete trial.
Remaining allocation: 1 + 1 = 2 complete trials.
"""

TRIALS = (
    {
        "success": False,
        "steps": [
            {"action": "search[grey sofa]", "observation": "Product A: sofa; details available."},
            {"action": "click[Product A]", "observation": "Product page with arm-style and colour options."},
            {"action": "click[Buy Now]", "observation": "Purchase complete. Task unsuccessful."},
        ],
    },
    {
        "success": True,
        "steps": [
            {"action": "search[grey sofa]", "observation": "Product A: sofa; details available."},
            {"action": "click[Product A]", "observation": "Product page with arm-style and colour options."},
            {"action": "click[Description]", "observation": "The product has the requested arm style."},
            {"action": "click[grey]", "observation": "Grey selected."},
            {"action": "click[Buy Now]", "observation": "Purchase complete. Task successful."},
        ],
    },
)


class FixtureClient:
    """Local API-shaped response fixture; never opens a network connection."""

    def __init__(self, method: str) -> None:
        self.method = method
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(deepcopy(kwargs))
        if self.method == "retry":
            raise AssertionError("Retry must not request a memory-model completion")
        content = JOINT_RESPONSE if self.method == "vex2" else REFLECTION
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=40, total_tokens=140),
            model=kwargs["model"],
            system_fingerprint="synthetic-api-walkthrough",
        )


def actor_messages(context: ActorContext) -> list[dict[str, str]]:
    """Use these positions in every actor call within the complete trial."""
    messages = [{"role": "system", "content": ACTOR_SYSTEM}]
    if context.system:
        messages.append({"role": "system", "content": context.system})
    prompt = f"Task: {TASK}\nObservation: {INITIAL_OBSERVATION}"
    if context.user:
        prompt = context.user + "\n\n" + prompt
    messages.append({"role": "user", "content": prompt})
    return messages


def walkthrough(name: str, *, show_requests: bool = False) -> dict[str, Any]:
    client = FixtureClient(name)
    method = load_method(name, client=client)
    method.reset_case(CaseConfig(
        total_trials=3,
        model="synthetic-fixture-model",
        task_scope=TASK,
        environment_rules=(
            "Use search[query] and click[visible label] on public pages. Each trial "
            "starts at the search page. Buy Now ends the trial; success ends the case."
        ),
        metadata={"benchmark": "webshop"},
    ))
    trials = []
    for index, fixed_trial in enumerate(TRIALS):
        context = method.context_for_trial(index)
        calls_after_read = len(client.calls)
        assert method.context_for_trial(index) == context
        assert len(client.calls) == calls_after_read, "Repeated reads must use the cached context"
        if index == 0:
            assert not context.system and not context.user
        elif name == "vex2":
            assert context.system == JOINT_RESPONSE, "VEX² must retain the raw joint response"
            assert not context.user
        elif name == "reflexion":
            assert REFLECTION in context.user and not context.system
        else:
            assert not context.system and not context.user

        # In your application, reset the environment and run the actor here.
        # These fixed traces are independent of the chosen method.
        messages = actor_messages(context)
        if name == "vex2" and index > 0:
            assert messages[1]["content"] == JOINT_RESPONSE
        trace = render_trajectory(
            task=TASK,
            initial_observation=INITIAL_OBSERVATION,
            steps=list(fixed_trial["steps"]),
            close_reason="success" if fixed_trial["success"] else "unsuccessful_purchase",
        )
        calls_before_observation = len(client.calls)
        method.observe_trial(TrialRecord(
            trial_index=index,
            trajectory=trace,
            success=bool(fixed_trial["success"]),
        ))
        trials.append({
            "trial_index": index,
            "synthetic_success": fixed_trial["success"],
            "context": {"system": context.system, "user": context.user},
            "actor_messages": messages,
            "trajectory": trace,
        })
        if fixed_trial["success"]:
            assert len(client.calls) == calls_before_observation, "Success must not trigger an update"
            break

    assert len(client.calls) == (0 if name == "retry" else 1)
    result = {
        "scope": "Synthetic API walkthrough; no task-performance claim or network calls.",
        "method": name,
        "method_completion_calls": len(client.calls),
        "trials": trials,
        "snapshot": method.snapshot(),
    }
    if show_requests:
        result["fixture_requests"] = client.calls
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("reflexion", "vex2", "retry"), default="reflexion")
    parser.add_argument("--show-requests", action="store_true")
    args = parser.parse_args()
    print(json.dumps(walkthrough(args.method, show_requests=args.show_requests), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
