"""Native SDK regressions for preserving the exact current environment state."""
from pathlib import Path
import base64
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
from test_interactive import FakeSession, PNG
from vrlbench.environments.interactive import Observation, TaskState, Transition

try:
    HAS_SDK = sys.version_info >= (3, 12) and importlib.util.find_spec('openhands.sdk') is not None
except ModuleNotFoundError:
    HAS_SDK = False


@unittest.skipUnless(HAS_SDK, 'requires native OpenHands SDK and Python 3.12+')
class ObservationContextTests(unittest.TestCase):
    def test_rejection_feedback_cannot_evict_latest_real_text_state(self):
        from openhands.sdk.context.view import View
        from openhands.sdk.event import ObservationEvent
        from openhands.sdk.llm import TextContent
        from vrlbench.actors.openhands import EnvironmentObservation
        from vrlbench.actors.observation_context import preserve_latest_context
        current='CURRENT_DOM_IDS '*1000
        events=[ObservationEvent(tool_name='environment_action',tool_call_id='call-0',action_id='action-0',
            observation=EnvironmentObservation(content=[TextContent(text=current)]))]
        for i in range(1,4):
            events.append(ObservationEvent(tool_name='environment_action',tool_call_id=f'call-{i}',action_id=f'action-{i}',
                observation=EnvironmentObservation.from_text('Rejected syntax '*200,is_error=True)))
        before=[e.model_dump_json() for e in events]
        projected=preserve_latest_context(llm=None,history_chars=128).condense(View(events=events))
        self.assertEqual(projected.events[0].observation.content[0].text,current)
        self.assertEqual([e.id for e in projected.events],[e.id for e in events])
        self.assertEqual([e.model_dump_json() for e in events],before)

    def test_missing_current_text_or_image_stops_before_usage_and_step_charge(self):
        from openhands.sdk.llm import Message, TextContent, ImageContent
        from vrlbench.actors.openhands import OpenHandsActor
        called = []
        observer = SimpleNamespace(before_request=lambda *a:called.append(a),
                                   on_response=lambda *a:None, on_error=lambda *a:None)
        with tempfile.TemporaryDirectory() as tmp:
            actor = OpenHandsActor(FakeSession(image=PNG), model='gpt-4.1-nano',
                max_environment_steps=4, artifact_dir=tmp, sdk_request_observer=observer)
            actor.prepare_trial()
            current = actor._sdk_content(actor.prepared.observation, 'Current state with fresh IDs')
            llm = SimpleNamespace(max_output_tokens=10, vision_is_active=lambda:True)
            tools = [SimpleNamespace(name='environment_action')]
            # Identical pixels do not make stale DOM text a current observation.
            stale = [Message(role='user',content=[TextContent(text='Old state'),current[1]])]
            with self.assertRaisesRegex(RuntimeError,'current observation text'):
                actor.before_sdk_request(llm,stale,tools)
            with self.assertRaisesRegex(RuntimeError,'current observation image'):
                actor.before_sdk_request(llm,[Message(role='user',content=[current[0]])],tools)
            with self.assertRaisesRegex(RuntimeError,'current observation image'):
                actor.before_sdk_request(llm,[Message(role='user',content=[current[0]]),
                                             Message(role='user',content=[current[1]])],tools)
            llm.vision_is_active=lambda:False
            with self.assertRaisesRegex(RuntimeError,'active vision'):
                actor.before_sdk_request(llm,[Message(role='user',content=current)],tools)
            self.assertEqual(called,[])
            self.assertEqual(actor.ledger.actor_turns,0)

    def test_serialization_loss_stops_before_request_admission(self):
        from openhands.sdk.llm import Message
        from vrlbench.actors.openhands import OpenHandsActor, _EnvironmentMessage
        called=[]
        observer=SimpleNamespace(before_request=lambda *a:called.append(a),
                                 on_response=lambda *a:None,on_error=lambda *a:None)
        with tempfile.TemporaryDirectory() as tmp:
            actor=OpenHandsActor(FakeSession(image=PNG),model='gpt-4.1-nano',
                max_environment_steps=4,artifact_dir=tmp,sdk_request_observer=observer)
            actor.prepare_trial()
            current=actor._sdk_content(actor.prepared.observation,'Current state '*7000)
            message=Message(role='tool',name='environment_action',tool_call_id='call-0',content=current)
            original=message.model_dump_json()
            llm=SimpleNamespace(max_output_tokens=10,vision_is_active=lambda:True,api_mode='responses')
            with patch.object(_EnvironmentMessage,'_maybe_truncate_tool_text',Message._maybe_truncate_tool_text):
                with self.assertRaisesRegex(RuntimeError,'serialization clipped'):
                    actor.before_sdk_request(llm,[message],[SimpleNamespace(name='environment_action')])
            self.assertEqual(called,[])
            self.assertEqual(actor.ledger.actor_turns,0)
            self.assertEqual(message.model_dump_json(),original)

    def test_real_sdk_long_history_keeps_current_frame_and_raw_events(self):
        self._run_sdk_long_history(api_mode='chat',visual=True,turns=12)

    def test_responses_http_preserves_complete_long_text_and_current_image(self):
        self._run_sdk_long_history(api_mode='responses',visual=True,turns=4)

    def test_string_chat_http_preserves_complete_long_tool_text(self):
        self._run_sdk_long_history(api_mode='chat',visual=False,turns=4,force_string=True)

    def _run_sdk_long_history(self,*,api_mode,visual,turns,force_string=False):
        from vrlbench.actors.openhands import OpenHandsActor
        requests=[]
        def strings(value):
            if isinstance(value,str):yield value
            elif isinstance(value,dict):
                for item in value.values():yield from strings(item)
            elif isinstance(value,list):
                for item in value:yield from strings(item)
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                index=len(requests);requests.append(body)
                expected = f'CURRENT_STAGE_{index}\n'+('page content '*6000)
                texts = list(strings(body))
                assert any(expected in text for text in texts), 'current text clipped on the HTTP wire'
                pictures=[text for text in texts if text.startswith('data:image/')]
                if visual:
                    assert 1<=len(pictures)<=2
                    assert base64.b64decode(pictures[-1].split(',',1)[1])==PNG+str(index).encode()
                else:assert not pictures
                reply={'id':f'chat-{index}','object':'chat.completion','created':1,'model':'gpt-4.1-nano',
                    'choices':[{'index':0,'finish_reason':'tool_calls','message':{'role':'assistant','content':None,
                    'tool_calls':[{'id':f'call-{index}','type':'function','function':{'name':'environment_action',
                    'arguments':json.dumps({'action':'continue'})}}]}}],
                    'usage':{'prompt_tokens':30,'completion_tokens':20,'total_tokens':50}}
                if api_mode=='responses':
                    reply={'id':f'resp-{index}','object':'response','created_at':1,'status':'completed',
                        'model':'gpt-4.1-nano','output':[{'id':f'fc-{index}','type':'function_call',
                            'call_id':f'call-{index}','name':'environment_action',
                            'arguments':json.dumps({'action':'continue'}),'status':'completed'}],
                        'usage':{'input_tokens':30,'output_tokens':20,'total_tokens':50}}
                data=json.dumps(reply).encode();self.send_response(200);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        class Session(FakeSession):
            def observation(self):return Observation(f'CURRENT_STAGE_{self.index}\n'+('page content '*6000),
                                                    PNG+str(self.index).encode() if visual else None)
            def reset(self,*,trial_index):
                self.index=0;return TaskState('Complete a long synthetic GUI task.',self.observation(),'fixed-case')
            def step(self,action):
                self.actions.append(action);self.index+=1
                return Transition(self.observation(),terminated=self.index==turns)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{
                'OPENAI_API_KEY':'local-test-only','LITELLM_LOCAL_MODEL_COST_MAP':'True',
                'OPENHANDS_SUPPRESS_BANNER':'1','OH_PERSISTENCE_DIR':str(Path(tmp)/'oh'),
                'VEX2_ALLOWED_RESPONSE_MODELS':'gpt-4.1-nano','VEX2_ALLOWED_RESPONSE_MODEL_PREFIXES':''}):
                session=Session(image=PNG)
                actor=OpenHandsActor(session,model='gpt-4.1-nano',max_environment_steps=turns,
                    artifact_dir=tmp,actor_params={'condenser':{
                        'entrypoint':'vrlbench.actors.observation_context:preserve_latest_context',
                        'kwargs':{'recent_observations':2,'history_chars':1000,'image_history':2}}},
                    sdk_model_config={'name':'gpt-4.1-nano','base_url':f'http://127.0.0.1:{server.server_port}/v1',
                        'timeout_seconds':5,'request_options':{'api_mode':api_mode,'force_string_serializer':force_string}})
                actor()
                self.assertEqual(len(requests),turns)
                self.assertEqual(len(session.actions),turns)
                raw=[json.loads(x) for x in actor.raw_event_path.read_text().splitlines()]
                observations=[x for x in raw if x.get('kind')=='ObservationEvent']
                # Projection must not mutate cached SDK events or evaluation artifacts.
                self.assertEqual(len(observations),turns)
                for e in observations:
                    text=e['observation']['content'][0]['text']
                    self.assertGreater(len(text),70000)
                    self.assertNotIn('Historical observation, superseded',text)
        finally:
            server.shutdown();server.server_close();thread.join(timeout=2)


if __name__=='__main__':unittest.main()
