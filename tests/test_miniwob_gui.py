"""Pixel controls cannot accept DOM selectors or leak HTML into observations."""
import unittest
import json
from types import SimpleNamespace
from vrlbench.environments.miniwob_gui import parse, PointerAction
from vrlbench.environments.native_sessions import MiniWoBSession
from test_interactive import PNG

class PixelGUI(unittest.TestCase):
    def test_strict_coordinates_and_selector_rejection(self):
        for value in ['{"action":"click","x":true,"y":1}',
                      '{"action":"click","x":640,"y":1}',
                      '{"action":"click","xpath":"//button"}',
                      '{"action":"type","text":"a","x":1}']:
            with self.assertRaises(ValueError):parse(value)
        self.assertEqual(parse('{"action":"click","x":10,"y":20}')['x'],10)

    def test_coordinate_click_executes_real_pointer_protocol(self):
        calls=[]
        PointerAction({'action':'click','x':10,'y':20})(SimpleNamespace(execute_cdp_cmd=lambda *a:calls.append(a)))
        self.assertEqual([c[1]['type'] for c in calls],['mouseMoved','mousePressed','mouseReleased'])
        self.assertEqual(calls[-1][1]['x'],10)

    def test_no_html_or_selector_leak_in_screenshot_mode(self):
        driver=SimpleNamespace(get_screenshot_as_png=lambda:PNG)
        env=SimpleNamespace(instance=SimpleNamespace(driver=driver))
        session=MiniWoBSession({'task':'enter-text','episode':0},{'observation_mode':'screenshot'},env=env)
        session._actions=object()
        obs=session._observation('<input id="SECRET_DOM_ID">')
        self.assertNotIn('SECRET_DOM_ID',obs.text)
        self.assertEqual(obs.image,PNG)
        self.assertEqual(session.source_identity['action_space'],'gui_json_v2')
        with self.assertRaises(ValueError):session.parse_response('agent.click_xpath("//input")')

    def test_drag_releases_at_destination(self):
        calls=[]
        PointerAction({'action':'drag','x':10,'y':20,'to_x':90,'to_y':30})(SimpleNamespace(execute_cdp_cmd=lambda *a:calls.append(a)))
        self.assertEqual((calls[-1][1]['x'],calls[-1][1]['y']),(90,30))
        self.assertTrue(all(c[1]['buttons']==1 for c in calls[2:-1]))

import importlib.util
@unittest.skipUnless(importlib.util.find_spec('openhands'), 'requires OpenHands SDK')
class GUIToolSchema(unittest.TestCase):
    def test_structured_tool_uses_native_json_and_rejects_bool_coordinates(self):
        from vrlbench.actors.openhands import GUIAction, EnvironmentExecutor
        with self.assertRaises(ValueError):GUIAction(action='click',x=True,y=2)
        seen=[]
        executor=EnvironmentExecutor(SimpleNamespace(_token_ledger=None,
            execute_environment_action=lambda raw:seen.append(json.loads(raw))))
        executor(GUIAction(action='click',x=10,y=20))
        self.assertEqual(seen,[{'action':'click','x':10,'y':20}])
        executor(GUIAction(action='drag',x=10,y=20,to_x=30,to_y=40,key='',text=''))
        self.assertEqual(seen[-1],{'action':'drag','x':10,'y':20,'to_x':30,'to_y':40})
        executor(GUIAction(action='type',text='',x=10,y=20))
        self.assertEqual(seen[-1],{'action':'type','text':'','x':10,'y':20})

class StopContract(unittest.TestCase):
    def test_stop_is_explicit_and_cannot_carry_a_claimed_reward(self):
        self.assertEqual(parse('{"action":"stop"}'),{'action':'stop'})
        with self.assertRaises(ValueError):parse('{"action":"stop","success":true}')


@unittest.skipUnless(importlib.util.find_spec('openhands'), 'requires OpenHands SDK')
class ToolDescription(unittest.TestCase):
    def test_gui_schema_and_description_agree(self):
        from vrlbench.actors.openhands import EnvironmentActionTool, GUIAction, _TRIAL_ACTORS
        actor=SimpleNamespace(session=SimpleNamespace(source_identity={'action_space':'gui_json_v2'},system_prompt='Screenshot controls'))
        _TRIAL_ACTORS['schema-test']=actor
        try:
            tool=EnvironmentActionTool.create(SimpleNamespace(id='schema-test'))[0]
            self.assertNotIn('native statement',tool.description)
            self.assertIn('structured fields',tool.description)
            self.assertEqual(tool.action_type,GUIAction)
        finally:_TRIAL_ACTORS.pop('schema-test')
