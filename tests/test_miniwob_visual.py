"""Native multimodal observations must follow reset and executed actions."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vrlbench import ModelSpec, SimulationSpec, WorkloadSpec
from vrlbench.environments.native_sessions import MiniWoBSession
from test_interactive import PNG


class NativeMultimodal(unittest.TestCase):
    def browser_retry_session(self, reset):
        env = SimpleNamespace(instance=SimpleNamespace(driver=SimpleNamespace(execute_script=lambda _: None)),
                              reset=reset, get_task=lambda: 'task', close=lambda: None)
        return MiniWoBSession({'task':'click-button','episode':4}, {}, env=env)

    def test_chrome_startup_retries_same_seed_and_records_attempts(self):
        error = type('SessionNotCreatedException', (Exception,),
                     {'__module__':'selenium.common.exceptions'})
        calls=[]
        def reset(**kw):
            calls.append(kw)
            if len(calls)<3: raise error('injected startup failure')
            return 'HTML'
        session=self.browser_retry_session(reset)
        with patch('vrlbench.environments.native_sessions.time.sleep'):
            session.reset(trial_index=0)
        self.assertEqual(calls,[{'seed':4}]*3)
        self.assertEqual(session.get_trial_metadata()['native_browser_start_attempts'],3)
        session.close()

    def test_chrome_startup_retry_is_bounded(self):
        error = type('SessionNotCreatedException', (Exception,),
                     {'__module__':'selenium.common.exceptions'})
        reset=unittest.mock.Mock(side_effect=error('injected startup failure'))
        session=self.browser_retry_session(reset)
        with patch('vrlbench.environments.native_sessions.time.sleep'), self.assertRaises(error):
            session.reset(trial_index=0)
        self.assertEqual(reset.call_count,3)
        self.assertEqual(session.get_trial_metadata()['native_browser_start_attempts'],3)
        session.close()

    def test_other_reset_failures_are_not_retried(self):
        reset=unittest.mock.Mock(side_effect=RuntimeError('invalid task'))
        session=self.browser_retry_session(reset)
        with self.assertRaisesRegex(RuntimeError,'invalid task'):
            session.reset(trial_index=0)
        self.assertEqual(reset.call_count,1)
        session.close()

    def test_native_raw_reward_survives_wall_clock_discount_and_resets(self):
        env = SimpleNamespace(instance=SimpleNamespace(driver=SimpleNamespace(execute_script=lambda _: None)),
                              reset=lambda **_: 'HTML', get_task=lambda: 'task', close=lambda: None)
        session = MiniWoBSession({'task':'click-button','episode':0}, {}, env=env)
        session.reset(trial_index=0)
        reward = env.instance.reward_processor({'raw_reward':1.0,'env_reward':0.0,'done':True})
        self.assertEqual(reward,1.0)
        self.assertEqual(session.get_trial_metadata()['native_discounted_reward'],0.0)
        self.assertEqual(session.source_identity['reward_mode'],'native_raw_action_budget')
        self.assertEqual(env.instance.reward_processor({'raw_reward':-1.0,'env_reward':-1.0,'done':True}),-1.0)
        session.reset(trial_index=1)
        self.assertNotIn('native_raw_reward',session.get_trial_metadata())
        session.close()

    def test_screenshot_inputs_and_reset_identity_use_native_state(self):
        driver = SimpleNamespace(execute_script=lambda _: None, get_screenshot_as_png=lambda: PNG)
        env = SimpleNamespace(instance=SimpleNamespace(driver=driver), reset=lambda **_: '<p>native HTML</p>',
                              get_task=lambda: 'native task', close=lambda: None)
        session = MiniWoBSession({'task':'click-button','episode':0}, {'include_screenshot':True}, env=env)
        state = session.reset(trial_index=0)
        self.assertEqual(state.observation.image, PNG)
        self.assertIn('1 x 1 pixels', state.observation.text)
        self.assertEqual(session.source_identity['observation_mode'], 'html_and_screenshot')
        def execute(_):
            session._actions.last_obs = 'new native HTML'
            session._actions.done = True
            session._actions.reward = 1.0
        with patch.object(session._actions, 'click_xpath', side_effect=execute):
            result = session.step('agent.click_xpath("//button")')
        self.assertEqual(result.observation.image, PNG)
        self.assertIn('new native HTML', result.observation.text)
        self.assertTrue(session.evaluate())
        session.reset(trial_index=1)
        self.assertFalse(session.evaluate())
        session.close()

    def test_text_only_still_works_without_screenshot_access(self):
        driver = SimpleNamespace(execute_script=lambda _: None)
        env = SimpleNamespace(instance=SimpleNamespace(driver=driver), reset=lambda **_: 'HTML',
                              get_task=lambda: 'task', close=lambda: None)
        session = MiniWoBSession({'task':'click-button','episode':0}, {}, env=env)
        self.assertIsNone(session.reset(trial_index=0).observation.image)
        session.close()

    def test_plan_rejects_invalid_vision_settings_and_strict_token_ceiling(self):
        base = SimulationSpec(WorkloadSpec('miniwob',[{'task':'click-button','episode':0}],
                                           params={'include_screenshot':True}), ModelSpec('vision'),
                              trials=2,max_environment_steps=2)
        base.validate()
        with self.assertRaisesRegex(ValueError, 'strict max_total_tokens'):
            replace(base,max_total_tokens=100).validate()
        with self.assertRaisesRegex(ValueError, 'boolean'):
            replace(base,workload=replace(base.workload,params={'include_screenshot':'yes'})).validate()
