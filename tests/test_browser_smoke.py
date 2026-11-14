"""Exercise real browser grading and trial isolation when GUI extras are installed."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('playwright'), 'install the GUI extra')
class BrowserSmokeIntegration(unittest.TestCase):
    def test_real_click_grading_and_trial_reset(self):
        from vrlbench.environments.browser_smoke import BrowserSession

        session = BrowserSession({'target': 'Save'}, {})
        self.addCleanup(session.close)
        first = session.reset(trial_index=0)
        self.assertTrue(first.observation.image.startswith(b'\x89PNG\r\n\x1a\n'))
        self.assertFalse(session.evaluate())
        # Locate the visible control only in this deterministic adapter test.
        # Online actors receive screenshots and must infer coordinates themselves.
        box = session.page.get_by_role('button', name='Save', exact=True).bounding_box()
        import json
        transition = session.step(json.dumps({'action': 'click',
            'x': int(box['x'] + box['width'] / 2),
            'y': int(box['y'] + box['height'] / 2)}))
        self.assertTrue(transition.terminated)
        self.assertTrue(session.evaluate())
        session.reset(trial_index=1)
        self.assertFalse(session.evaluate(), 'a previous successful trial must not leak')
        self.assertFalse(session.step('{"action":"click","x":true,"y":1}').terminated)
        self.assertFalse(session.evaluate())
