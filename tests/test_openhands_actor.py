"""One native OpenHands SDK regression, using a local HTTP model endpoint."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from test_interactive import FakeSession, PNG
from vrlbench.runtime.er.base import ERSlot
from vrlbench.runtime.loop import run_episode

try:
    HAS_SDK = sys.version_info >= (3, 12) and importlib.util.find_spec("openhands.sdk") is not None
except ModuleNotFoundError:
    HAS_SDK = False


@unittest.skipUnless(HAS_SDK, "OpenHands actor requires the optional SDK and Python 3.12+")
class OpenHandsActorTests(unittest.TestCase):
    def test_real_sdk_trace_becomes_memory_in_a_fresh_trial(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                index = len(requests)
                if index == 1:
                    message, finish = {"role": "assistant", "content": "Finished without acting."}, "stop"
                else:
                    action = "wrong" if index == 2 else "right"
                    message = {"role": "assistant", "content": "Use the visible interface.",
                               "tool_calls": [{"id": f"call-{index}", "type": "function",
                                               "function": {"name": "environment_action",
                                                            "arguments": json.dumps({"action": action})}}]}
                    finish = "tool_calls"
                response = {"id": f"chatcmpl-local-{index}", "object": "chat.completion",
                            "created": 1, "model": "gpt-4.1-nano",
                            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                            "usage": {"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50}}
                data = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        class Session(FakeSession):
            def step(self, action):
                self.outcome = action == "right"
                return super().step(action)

        class TraceMemory(ERSlot):
            def reset(self):
                self.memory, self.trace = "", None

            def build_section(self):
                return self.memory

            def read_placement(self):
                return "user_prefix"

            def on_trial_end(self, trajectory, *, actor_trajectory=None, success=False,
                             metadata=None, **_kwargs):
                if (success or not metadata or not metadata.get("has_future_trial")
                        or not metadata.get("policy_evidence_eligible")):
                    return
                self.trace = json.loads(actor_trajectory)
                events = self.trace["events"]
                failed_action = next(event["action"]["action"] for event in events
                                     if event["kind"] == "ActionEvent"
                                     and event["tool_name"] == "environment_action")
                feedback = next(part["text"] for event in events
                                if event["kind"] == "ObservationEvent"
                                for part in event["observation"]["content"]
                                if part["type"] == "text").splitlines()[0]
                self.memory = f"Recorded failure: {failed_action}; {feedback}. Choose right next time."

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
                "OPENAI_API_KEY": "local-test-only", "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                "OH_PERSISTENCE_DIR": str(Path(temp) / "sdk-state"),
            }):
                os.environ.pop("VEX2_ALLOWED_RESPONSE_MODELS", None)
                from vrlbench.actors.openhands import OpenHandsActor

                session, memory = Session(done=True, image=PNG), TraceMemory()
                actor = OpenHandsActor(session, model="gpt-4.1-nano", max_environment_steps=3,
                    artifact_dir=temp, sdk_model_config={"name": "gpt-4.1-nano",
                    "base_url": f"http://127.0.0.1:{server.server_port}/v1", "timeout_seconds": 5})
                result = run_episode(actor, lambda _trace: actor.success, memory,
                    "gpt-4.1-nano", 3, metadata={"benchmark": "webarena", "case_id": "sdk-local"})

                self.assertEqual([trial.success for trial in result.trials], [False, True])
                self.assertEqual(session.resets, [0, 1])
                self.assertEqual(session.actions, ["wrong", "right"])
                self.assertEqual(len(requests), 3)  # no request after environment success
                self.assertEqual(result.trials[0].trial_metadata["action_attempts"], 2)
                self.assertIn("wrong", json.dumps(memory.trace))
                self.assertIn("Visible feedback", json.dumps(memory.trace))
                self.assertNotIn("data:image", json.dumps(memory.trace))
                self.assertTrue(any(message["role"] == "user" and memory.memory in json.dumps(message)
                                    for message in requests[-1]["messages"]))
                self.assertFalse(any(message["role"] == "tool" for message in requests[-1]["messages"]))
                self.assertEqual(result.resource_usage["api_calls"], 3)
                self.assertEqual(result.resource_usage["total_tokens"], 150)
                kinds = set()
                for trial in result.trials:
                    source = trial.trial_metadata["sdk_event_log"]
                    data = Path(source["path"]).read_bytes()
                    self.assertEqual(hashlib.sha256(data).hexdigest(), source["sha256"])
                    kinds.update(json.loads(line)["kind"] for line in data.splitlines())
                self.assertTrue({"ActionEvent", "ObservationEvent"} <= kinds)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
