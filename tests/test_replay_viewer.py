"""Saved-run replay is offline, faithful to recorded time, and inert."""
from __future__ import annotations

import base64
import hashlib
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vrlbench.replay_viewer import export_replay, replay_data, static_page_html


def trace(initial="initial", action="click[item]", observation="detail"):
    return "\n".join(["policy_ao_trace", "Task: Find the item",
        "Initial observation JSON: " + json.dumps(initial),
        "AO 1 JSON: " + json.dumps({"action": action, "observation": observation}),
        "Terminal JSON: " + json.dumps({"trial_close_reason": "ao_horizon"})])


def invalid_event(position, before, raw="not an action", reason="No valid action was returned"):
    return {"after_environment_step": position, "raw_response": raw, "reason": reason,
            "budget_used_before": before, "budget_used_after": before + 1}


class ReplayViewerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.run = self.root / "reflexion"
        self.run.mkdir()
        self.payload = {"status": "complete", "trials": 6, "counts": {"planned": 1, "completed": 1},
            "output_dir": str(self.run),
            "config": {"model": {"name": "saved-model"}, "memory": {"name": "reflexion"},
                       "max_environment_steps": 15},
            "cases": [{"case_id": "sample_0000", "task": "webshop", "trials": [
                {"trial": 0, "success": False, "context": "", "system_context": "", "user_context": "",
                 "policy_trajectory": trace(), "trial_metadata": {"environment_resets": 1}},
                {"trial": 1, "success": False, "context": "try a different item", "system_context": "",
                 "user_context": "try a different item", "policy_trajectory": trace(),
                 "controller_state_pre": {"before": 1}, "controller_state": {"after": 2}}]}]}
        self.save()

    def save(self):
        self.path = self.run / "evaluation.json"
        self.path.write_text(json.dumps(self.payload), encoding="utf-8")

    def set_invalid_events(self, events, *, native=1, budget=6):
        row = self.payload["cases"][0]["trials"][0]
        row["trial_metadata"].update(budget_unit="environment_action_or_invalid_response",
            max_environment_steps=budget, environment_steps=native,
            action_attempts=native + len(events), invalid_action_attempts=len(events),
            remaining_environment_steps=budget - native - len(events), invalid_action_events=events)
        return row

    def viewer_positions(self, requests, *, html=None):
        """Exercise the local viewer with a tiny DOM stub, not a browser."""
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js is not available for the offline viewer state check")
        output = (html if html is not None else
                  export_replay([self.run], self.root / "ui.html").read_text(encoding="utf-8"))
        embedded = output.split('id="replay-data">', 1)[1].split("</script>", 1)[0]
        script = output.split('<script>\n', 1)[1].split("</script>", 1)[0]
        harness = r'''
const input=JSON.parse(require('fs').readFileSync(0,'utf8'));
class Element {
  constructor(tag=''){this.tagName=tag.toUpperCase();this.children=[];this._text='';this.events={};this.style={};this.attrs={};this.hidden=false;
    this.classList={add(){},remove(){},toggle(){}};}
  set textContent(value){this._text=String(value);this.children=[];}
  get textContent(){return this._text+this.children.map(c=>c.textContent).join('');}
  appendChild(child){this.children.push(child);return child;}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this._text='';this.children=children;}
  setAttribute(key,value){this.attrs[key]=value;}
  addEventListener(name,fn){this.events[name]=fn;}
}
const elements={};const get=id=>elements[id]||(elements[id]=new Element());
global.document={getElementById:get,createElement:tag=>new Element(tag),hidden:false,
  querySelectorAll:()=>['phase-observe','phase-act','phase-feedback','phase-end'].map(get),addEventListener(){}};
get('replay-data').textContent=JSON.stringify(input.data);get('speed').value='1600';
eval(input.script);
function snapshot(){return {states:Object.fromEntries(get('evaluation-state').children.map(r=>r.children.map(c=>c.textContent))),
  heading:get('action-heading').textContent,action:get('action-code').textContent,feedback:get('feedback-text').textContent,
  outcome_hidden:get('outcome').hidden,memory_after_hidden:get('memory-state-after-wrap').hidden,
  metadata:get('metadata').textContent,error:get('error').textContent,
  images:get('observation').children.filter(child=>child.tagName==='IMG').map(child=>child.src),
  mode:get('observation-mode').textContent};}
const out=[snapshot()];
for(const request of input.requests){
  if(request.trial!==undefined)get('timeline').children[request.trial].children[0].events.click();
  else get('progress').events.input({target:{value:String(request.position)}});
  out.push(snapshot());
}
process.stdout.write(JSON.stringify(out));
'''
        result = subprocess.run([node, "-e", harness], input=json.dumps({"data": json.loads(embedded),
            "script": script, "requests": requests}), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        snapshots = json.loads(result.stdout)
        self.assertTrue(all(not item["error"] for item in snapshots), snapshots)
        return snapshots

    def test_rejected_response_precedes_native_action_without_environment_transition(self):
        raw = b"\x89PNG\r\n\x1a\nreset"
        image = self.run / "reset.png"
        image.write_bytes(raw)
        row = self.set_invalid_events([invalid_event(0, 0, raw="I cannot decide yet")])
        row["trial_metadata"]["replay_observations"] = [{"observation_index": 0, "kind": "image",
            "path": str(image), "sha256": hashlib.sha256(raw).hexdigest()}]
        self.save()
        before = {p.name: p.read_bytes() for p in self.run.iterdir()}
        trial = replay_data([self.run])["runs"][0]["cases"][0]["trials"][0]
        rejected, native = trial["frames"]
        self.assertEqual(rejected["kind"], "invalid_action")
        self.assertIsNone(rejected["action"])
        self.assertEqual(rejected["raw_response"], "I cannot decide yet")
        for field in ("observation", "image", "html", "snapshot_notice"):
            self.assertEqual(rejected["before_" + field], rejected["after_" + field])
        self.assertEqual(rejected["before_observation"], "initial")
        self.assertTrue(rejected["before_image"].startswith("data:image/png;base64,"))
        self.assertEqual((rejected["environment_steps_before"], rejected["environment_steps_after"]), (0, 0))
        self.assertEqual((native["step_index"], native["budget_used_before"], native["budget_used_after"]), (0, 1, 2))
        self.assertEqual((native["environment_steps_before"], native["environment_steps_after"]), (0, 1))
        export_replay([self.run], self.root / "rejected.html")
        self.assertEqual({p.name: p.read_bytes() for p in self.run.iterdir()}, before)

    def test_all_rejected_responses_keep_zero_native_actions_and_recorded_empty_raw_reply(self):
        events = [invalid_event(0, 0, raw=""), invalid_event(0, 1, raw="still no action")]
        row = self.set_invalid_events(events, native=0, budget=2)
        row["policy_trajectory"] = '\n'.join(['policy_ao_trace', 'Task: Find the item',
            'Initial observation JSON: "actual reset page"',
            'Terminal JSON: {"trial_close_reason":"ao_horizon"}'])
        self.save()
        trial = replay_data([self.run])["runs"][0]["cases"][0]["trials"][0]
        self.assertTrue(trial["has_action_record"])
        self.assertEqual([f["raw_response"] for f in trial["frames"]], ["", "still no action"])
        self.assertEqual([f["budget_used_after"] for f in trial["frames"]], [1, 2])
        self.assertTrue(all(f["before_observation"] == f["after_observation"] == "actual reset page"
                            and f["environment_steps_after"] == 0 for f in trial["frames"]))
        self.assertEqual(trial["metadata"]["remaining_environment_steps"], 0)

    def test_mixed_events_preserve_order_and_native_step_budget_prefixes(self):
        row = self.set_invalid_events([invalid_event(0, 0, "first"), invalid_event(1, 2, "second"),
            invalid_event(1, 3, "third"), invalid_event(2, 5, "last")], native=2)
        row["policy_trajectory"] = '\n'.join(['policy_ao_trace', 'Task: Find the item',
            'Initial observation JSON: "initial"',
            'AO 1 JSON: {"action":"click[item]","observation":"detail"}',
            'AO 2 JSON: {"action":"click[buy]","observation":"result"}',
            'Terminal JSON: {"trial_close_reason":"ao_horizon"}'])
        self.save()
        trial = replay_data([self.run])["runs"][0]["cases"][0]["trials"][0]
        frames = trial["frames"]
        self.assertEqual([f.get("raw_response", f["action"]) for f in frames],
                         ["first", "click[item]", "second", "third", "click[buy]", "last"])
        self.assertEqual([f["budget_used_before"] for f in frames], list(range(6)))
        self.assertEqual([f["budget_used_after"] for f in frames], list(range(1, 7)))
        self.assertEqual([f["environment_steps_after"] for f in frames], [0, 1, 1, 1, 2, 2])
        self.assertEqual([f["step_index"] for f in frames if f["kind"] == "environment_action"], [0, 1])
        self.assertEqual(frames[-1]["before_observation"], "result")
        self.assertEqual(trial["metadata"]["action_attempts"], 6)
        self.assertEqual(trial["metadata"]["environment_steps"], 2)
        self.assertEqual(trial["metadata"]["remaining_environment_steps"], 0)

    def test_old_actor_prose_does_not_create_rejected_events_or_new_budget_policy(self):
        row = self.payload["cases"][0]["trials"][0]
        row["actor_trajectory"] = "Thought: invalid response\nAction: click[item]"
        row["trial_metadata"]["invalid_actor_responses"] = 1
        self.save()
        trial = replay_data([self.run])["runs"][0]["cases"][0]["trials"][0]
        self.assertEqual(len(trial["frames"]), 1)
        self.assertNotIn("budget_used_before", trial["frames"][0])
        self.assertNotIn("invalid_action_events", trial["metadata"])

    def test_rejected_event_anchor_and_budget_must_match_canonical_native_steps(self):
        for event in (invalid_event(2, 2), invalid_event(0, 1)):
            self.set_invalid_events([event])
            self.save()
            with self.assertRaisesRegex(ValueError, "invalid-action event position or budget"):
                replay_data([self.run])

    def test_viewer_positions_separate_trial_action_and_native_step_budgets(self):
        row = self.set_invalid_events([invalid_event(0, 0, raw="</script><script>unexpected()</script>",
            reason="No supported action")], budget=2)
        row["memory_state"] = {"updated": True}
        self.payload["cases"][0]["trials"] = [row]
        self.save()
        start, rejected, native_before, end = self.viewer_positions([
            {"position": 1}, {"position": 2}, {"position": 3}])
        self.assertEqual(start["heading"], "Rejected actor response")
        self.assertEqual(start["action"], row["trial_metadata"]["invalid_action_events"][0]["raw_response"])
        self.assertEqual(start["states"]["Trials used"], "1 (current trial included)")
        self.assertEqual(start["states"]["Trials remaining"], "5")
        self.assertEqual([s["states"]["Action budget used"] for s in (start, rejected, native_before, end)],
                         ["0", "1", "1", "2"])
        self.assertEqual([s["states"]["Action budget remaining"] for s in (start, rejected, native_before, end)],
                         ["2", "1", "1", "0"])
        self.assertEqual([s["states"]["Environment steps"] for s in (start, rejected, native_before, end)],
                         ["0", "0", "0", "1"])
        self.assertIn("Environment not executed", rejected["feedback"])
        self.assertIn("No supported action", rejected["feedback"])
        self.assertTrue(all(s["outcome_hidden"] and s["memory_after_hidden"] for s in (start, rejected, native_before)))
        self.assertFalse(end["outcome_hidden"])
        self.assertFalse(end["memory_after_hidden"])
        self.assertNotIn("invalid_action_events", start["metadata"])
        self.assertIn("invalid_action_events", end["metadata"])

    def test_viewer_all_invalid_final_position_and_unknown_trial_remain_distinct(self):
        row = self.set_invalid_events([invalid_event(0, 0, raw=""), invalid_event(0, 1)], native=0, budget=2)
        row["policy_trajectory"] = '\n'.join(['policy_ao_trace', 'Task: Find the item',
            'Initial observation JSON: "initial"', 'Terminal JSON: {"trial_close_reason":"ao_horizon"}'])
        unknown = self.payload["cases"][0]["trials"][1]
        unknown.pop("policy_trajectory")
        self.save()
        start, final_before, end, summary = self.viewer_positions([
            {"position": 2}, {"position": 3}, {"trial": 1}])
        self.assertEqual(start["action"], "Empty (recorded)")
        self.assertTrue(final_before["outcome_hidden"])
        self.assertFalse(end["outcome_hidden"])
        self.assertEqual(end["states"]["Environment steps"], "0")
        self.assertEqual(end["states"]["Action budget used"], "2")
        self.assertEqual(summary["heading"], "Selected action")
        self.assertEqual(summary["states"]["Action count"], "Not recorded")
        self.assertEqual(summary["states"]["Trials remaining"], "4")
        self.assertNotIn("Budget rule", summary["states"])

    def test_steps_memory_and_controller_state_remain_distinct(self):
        before = self.path.read_bytes()
        data = replay_data([self.run])
        run = data["runs"][0]
        self.assertEqual(run["trial_budget"], 6)
        trials = run["cases"][0]["trials"]
        self.assertEqual(trials[0]["memory"]["user_context"], "")
        self.assertEqual(trials[1]["memory"]["user_context"], "try a different item")
        self.assertEqual(trials[1]["memory"]["controller_state_pre"], {"before": 1})
        frame = trials[0]["frames"][0]
        self.assertEqual((frame["before_observation"], frame["action"], frame["after_observation"]),
                         ("initial", "click[item]", "detail"))
        self.assertIsNone(frame["before_image"])
        self.assertEqual(self.path.read_bytes(), before)

    def test_memory_store_snapshots_are_not_inferred_from_context(self):
        row = self.payload["cases"][0]["trials"][1]
        row["memory_state_pre"] = {"entries": ["recorded item"]}
        row["memory_state"] = {"entries": ["recorded item", "updated item"]}
        self.save()
        memory = replay_data([self.run])["runs"][0]["cases"][0]["trials"][1]["memory"]
        self.assertEqual(memory["state_pre"], row["memory_state_pre"])
        self.assertEqual(memory["state_after"], row["memory_state"])
        self.assertNotEqual(memory["state_pre"], memory["user_context"])

    def test_images_require_valid_digest_and_are_embedded(self):
        raw = b"\x89PNG\r\n\x1a\nexample"
        image = self.run / "image.png"
        image.write_bytes(raw)
        meta = self.set_invalid_events([invalid_event(0, 0)])["trial_metadata"]
        meta["replay_observations"] = [{"observation_index": 0, "kind": "image", "path": str(image),
                                       "sha256": hashlib.sha256(raw).hexdigest()}]
        self.save()
        frame = replay_data([self.run])["runs"][0]["cases"][0]["trials"][0]["frames"][0]
        self.assertTrue(frame["before_image"].startswith("data:image/png;base64,"))
        image.write_bytes(b"changed")
        frame = replay_data([self.run])["runs"][0]["cases"][0]["trials"][0]["frames"][0]
        self.assertIsNone(frame["before_image"])
        self.assertIn("checksum", frame["before_snapshot_notice"])
        self.assertEqual(frame["before_snapshot_notice"], frame["after_snapshot_notice"])

    def test_external_images_are_deduplicated_raw_relative_and_exclusive(self):
        raw = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRskAAAAASUVORK5CYII="
        )
        digest = hashlib.sha256(raw).hexdigest()
        image = self.run / "reset.png"
        image.write_bytes(raw)
        duplicate = self.run / "after.png"
        duplicate.write_bytes(raw)
        literal = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
        row = self.payload["cases"][0]["trials"][0]
        row["policy_trajectory"] = trace(initial=literal)
        row["trial_metadata"]["replay_observations"] = [
            {"observation_index": index, "kind": "image", "path": str(path), "sha256": digest}
            for index, path in enumerate((image, duplicate))]
        self.save()

        output = self.root / "visual #1%.html"
        assets = self.root / "visual #1%.assets"
        export_replay([self.run], output, external_images=True)
        html = output.read_text(encoding="utf-8")
        data = json.loads(html.split('id="replay-data">', 1)[1].split("</script>", 1)[0])
        trial = data["runs"][0]["cases"][0]["trials"][0]
        expected = "visual%20%231%25.assets/" + digest + ".png"
        self.assertEqual(trial["initial_image"], expected)
        self.assertEqual(trial["frames"][0]["before_image"], expected)
        self.assertEqual(trial["frames"][0]["after_image"], expected)
        self.assertEqual(trial["initial_observation"], literal)
        self.assertEqual(trial["frames"][0]["action"], "click[item]")
        self.assertEqual([path.name for path in assets.iterdir()], [digest + ".png"])
        self.assertEqual((assets / (digest + ".png")).read_bytes(), raw)
        self.assertIn('content="img-src \'self\' data:"', html)
        shown = self.viewer_positions([], html=html)[0]
        self.assertEqual(shown["images"], [expected])
        self.assertEqual(shown["mode"], "Recorded screenshot")
        embedded = html.split('id="replay-data">', 1)[1].split("</script>", 1)[0]
        for invalid in ("https://example.org/x.assets/", "../outside.assets/",
                        "..%2Foutside.assets/", "nested%2Foutside.assets/",
                        "nested%5Coutside.assets/"):
            bad = invalid + digest + ".png"
            trial["initial_image"] = bad
            trial["frames"][0]["before_image"] = bad
            trial["frames"][0]["after_image"] = bad
            untrusted = html.replace(embedded, json.dumps(data), 1)
            with self.subTest(image=bad):
                rejected = self.viewer_positions([], html=untrusted)[0]
                self.assertEqual(rejected["images"], [])
                self.assertEqual(rejected["mode"], "Recorded text observation")
        with self.assertRaises(FileExistsError):
            export_replay([self.run], output, external_images=True)
        self.assertEqual(output.read_text(encoding="utf-8"), html)
        self.assertEqual((assets / (digest + ".png")).read_bytes(), raw)

        blocked = self.root / "blocked.html"
        existing = self.root / "blocked.assets"
        existing.mkdir()
        sentinel = existing / "keep"
        sentinel.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            export_replay([self.run], blocked, external_images=True)
        self.assertFalse(blocked.exists())
        self.assertEqual(sentinel.read_bytes(), b"existing")

    def test_shared_trial_snapshots_require_the_explicit_source_run_even_after_moving(self):
        source = self.root / "retry"
        source.mkdir()
        raw = b"\x89PNG\r\n\x1a\nshared trial"
        (source / "page.png").write_bytes(raw)
        payload = deepcopy(self.payload)
        payload["output_dir"] = "/original/retry"
        (source / "evaluation.json").write_text(json.dumps(payload))
        self.payload["cases"][0]["trials"][0]["trial_metadata"]["replay_observations"] = [
            {"observation_index": 0, "kind": "image", "path": "/original/retry/page.png",
             "sha256": hashlib.sha256(raw).hexdigest()}]
        self.save()
        def first_frame(paths):
            return replay_data(paths)["runs"][0]["cases"][0]["trials"][0]["frames"][0]
        self.assertIsNone(first_frame([self.run])["before_image"])
        self.assertIn("source run", first_frame([self.run])["before_snapshot_notice"])
        self.assertTrue(first_frame([self.run, source])["before_image"].startswith("data:image/png;base64,"))

    def test_snapshot_paths_cannot_escape_run_including_symlinks(self):
        outside = self.root / "secret.html"
        outside.write_text("not a run artifact")
        link = self.run / "link.html"
        link.symlink_to(outside)
        meta = self.set_invalid_events([invalid_event(0, 0)])["trial_metadata"]
        meta["replay_observations"] = [{"observation_index": 0, "kind": "html", "path": str(link),
                                       "sha256": hashlib.sha256(outside.read_bytes()).hexdigest()}]
        self.save()
        frame = replay_data([self.run])["runs"][0]["cases"][0]["trials"][0]["frames"][0]
        self.assertIn("outside", frame["before_snapshot_notice"])
        self.assertIsNone(frame["before_html"])
        self.assertIsNone(frame["after_html"])

    def test_static_html_has_no_scripts_remote_resources_or_form_actions(self):
        html = static_page_html('<script>alert(1)</script><img src="https://example.org/x">'
                                '<form action="https://example.org"><button onclick="alert(2)">Buy</button></form>'
                                '<a href="javascript:alert(3)">link</a><p>Real page text</p>')
        self.assertNotIn("<script", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("onclick", html)
        self.assertNotIn("javascript:", html)
        self.assertIn("Real page text", html)
        self.assertIn('disabled=""', html)

    def test_json_script_breakout_is_escaped_and_export_is_exclusive(self):
        value = '</script><script>alert("test")</script>'
        self.payload["cases"][0]["trials"][0]["user_context"] = value
        self.save()
        output = export_replay([self.run], self.root / "replay.html")
        html = output.read_text()
        self.assertNotIn(value, html)
        self.assertIn("\\u003c/script\\u003e", html)
        embedded = html.split('id="replay-data">', 1)[1].split("</script>", 1)[0]
        self.assertEqual(json.loads(embedded)["runs"][0]["cases"][0]["trials"][0]["memory"]["user_context"], value)
        with self.assertRaises(FileExistsError):
            export_replay([self.run], output)

    def test_missing_trace_is_not_reconstructed_and_unknown_differs_from_empty(self):
        row = self.payload["cases"][0]["trials"][0]
        row.pop("policy_trajectory")
        row.pop("system_context")
        self.payload["status"] = "timeout"
        self.save()
        trial = replay_data([self.run])["runs"][0]["cases"][0]["trials"][0]
        self.assertEqual(trial["frames"], [])
        self.assertFalse(trial["has_action_record"])
        self.assertIsNone(trial["memory"]["system_context"])
        self.assertEqual(trial["memory"]["user_context"], "")
        with self.assertRaisesRegex(ValueError, "not recorded"):
            replay_data([self.run], case_id="missing")

    def test_zero_action_trial_retains_initial_observation_and_end_reason(self):
        row = self.payload["cases"][0]["trials"][0]
        row["policy_trajectory"] = '\n'.join([
            'policy_ao_trace', 'Task: Find the item',
            'Initial observation JSON: "real reset page"',
            'Terminal JSON: {"trial_close_reason":"actor_protocol_failure"}'])
        self.save()
        trial = replay_data([self.run])["runs"][0]["cases"][0]["trials"][0]
        self.assertEqual(trial["frames"], [])
        self.assertTrue(trial["has_action_record"])
        self.assertEqual(trial["initial_observation"], "real reset page")
        self.assertEqual(trial["close_reason"], "actor_protocol_failure")

    def test_cli_does_not_import_model_runtime_or_load_plugins(self):
        self.payload["config"]["memory"]["factory"] = {"entrypoint": "missing_module:create"}
        self.save()
        output = self.root / "replay.html"
        script = ("import sys\n"
                  "from vrlbench.cli import main\n"
                  f"assert main(['replay', {str(self.run)!r}, '--output', {str(output)!r}, '--external-images']) == 0\n"
                  "assert 'openai' not in sys.modules\n"
                  "assert 'vrlbench.runtime' not in sys.modules\n")
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
        env.pop("OPENAI_API_KEY", None)
        result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
