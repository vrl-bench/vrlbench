"""Small real-browser GUI integration check with native screenshot observations."""

import hashlib
from importlib.metadata import version
import json

from vrlbench.environments.interactive import Observation, TaskState, Transition, run_cases


PAGE = """<!doctype html><html><head><style>
body {margin:0;background:#f3f5f8;font:24px Arial;color:#172b3e}
main {margin:120px auto;width:700px;padding:40px;background:white;border-radius:16px}
button {font:22px Arial;padding:24px 50px;margin:20px;border:0;border-radius:8px}
button:first-of-type {background:#1d66c2;color:white}
button:last-of-type {background:#e2e6eb;color:#172b3e}
</style></head><body><main><h1>Document settings</h1><p>Choose an action.</p>
<button>Save</button><button>Cancel</button><p id="status">Waiting for input</p>
</main><script>document.querySelectorAll('button').forEach(button => {
button.onclick=()=>{document.body.dataset.choice=button.textContent;
document.getElementById('status').textContent='Selected '+button.textContent;};
});</script></body></html>"""


class BrowserSession:
    task_group = "browser_smoke"
    system_prompt = ('Use the screenshot to operate the browser. Return a JSON object '
                     '{"action":"click","x":100,"y":200} with absolute pixel coordinates '
                     'in the 960 by 640 screenshot. Exactly one click per action.')
    basic_rules = "Every trial resets the browser. The page verifies the selected button."
    task_scope = "Click the requested visible button."

    def __init__(self, case, params):
        from playwright.sync_api import sync_playwright

        self.target = case["target"]
        if self.target not in {"Save", "Cancel"}:
            raise ValueError("browser smoke target must be Save or Cancel")
        self.source_identity = {"html_sha256": hashlib.sha256(PAGE.encode()).hexdigest(),
                                "target": self.target, "playwright": version("playwright"),
                                "viewport": [960, 640]}
        self.engine = sync_playwright().start()
        self.context = None
        try:
            self.browser = self.engine.chromium.launch(headless=True)
            self.source_identity["browser"] = self.browser.version
        except BaseException:
            self.engine.stop()
            raise

    def observation(self, feedback=""):
        return Observation(text="Screenshot dimensions: 960 x 640 pixels. " + feedback,
                           image=self.page.screenshot(type="png"))

    def reset(self, *, trial_index):
        if self.context is not None:
            self.context.close()
        self.context = self.browser.new_context(viewport={"width": 960, "height": 640})
        self.context.route("**/*", lambda route: route.abort())
        self.page = self.context.new_page()
        self.page.set_content(PAGE)
        return TaskState(task=f'Click the "{self.target}" button.',
                         observation=self.observation(),
                         identity=json.dumps(self.source_identity, sort_keys=True))

    def step(self, raw):
        try:
            action = json.loads(raw)
            if (not isinstance(action, dict) or set(action) != {"action", "x", "y"}
                    or action["action"] != "click"
                    or type(action["x"]) is not int or type(action["y"]) is not int
                    or not 0 <= action["x"] < 960 or not 0 <= action["y"] < 640):
                raise ValueError("Invalid click coordinates")
        except (ValueError, TypeError):
            return Transition(self.observation("Invalid action. " + self.system_prompt))
        self.page.mouse.click(action["x"], action["y"])
        selected = self.page.get_attribute("body", "data-choice")
        return Transition(self.observation(), terminated=selected is not None)

    def evaluate(self):
        return self.page.get_attribute("body", "data-choice") == self.target

    def close(self):
        try:
            self.browser.close()
        finally:
            self.engine.stop()


class Backend:
    def run(self, spec, runtime):
        return run_cases(spec, runtime, BrowserSession)


def create_backend():
    return Backend()


def example_spec():
    from vrlbench.specs import FactorySpec, ModelSpec, SimulationSpec, WorkloadSpec

    return SimulationSpec(
        WorkloadSpec("browser_smoke", [{"case_id": "save", "target": "Save"},
                                       {"case_id": "cancel", "target": "Cancel"}],
                     params={"capture_observations": True, "uses_model": True,
                             "requires_vision": True},
                     factory=FactorySpec("vrlbench.environments.browser_smoke:create_backend")),
        ModelSpec("YOUR_VISION_MODEL_NAME"), trials=2, max_environment_steps=4,
        timeout_seconds=300, require_provider_usage=True, output_dir="runs/gui_smoke")
