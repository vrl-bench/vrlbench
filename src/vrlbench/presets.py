"""Small, explicit native task selections shipped with installed distributions."""
from .specs import ActorSpec, ModelSpec, SimulationSpec, WorkloadSpec

# Version 1: changing this selection requires a new preset name/version.
MINIWOB_VISUAL_TASKS = ('click-button', 'click-link', 'enter-text', 'click-checkboxes',
                       'click-option', 'focus-text', 'guess-number', 'use-slider')


def miniwob_visual():
    return SimulationSpec(
        WorkloadSpec('miniwob', [{'task': task, 'episode': seed}
            for task in MINIWOB_VISUAL_TASKS for seed in range(2)], params={
                'headless': True, 'trial_seed_mode': 'paired_trial',
                'capture_observations': True, 'include_screenshot': True}),
        ModelSpec('YOUR_VISION_MODEL_NAME'),
        actor=ActorSpec('openhands', {'max_output_tokens': 1024}),
        trials=2, max_environment_steps=12, require_provider_usage=True,
        timeout_seconds=600, output_dir='runs/miniwob_visual')


def miniwob_gui():
    from dataclasses import replace
    base = miniwob_visual()
    return replace(base, workload=replace(base.workload,
        params={**base.workload.params,'observation_mode':'screenshot'}),
        max_environment_steps=20, output_dir='runs/miniwob_gui')
