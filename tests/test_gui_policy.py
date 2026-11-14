"""GUI admission preserves Python aliases and the submitted action source."""

import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from vrlbench.environments.gui_policy import validate_pyautogui_program


class GUIActionPolicyTests(unittest.TestCase):
    def test_alias_and_canonical_programs_execute_the_same_calls_without_rewriting(self):
        programs = [
            "import pyautogui\nimport time\npyautogui.click(405,75)\ntime.sleep(0.8)",
            "import pyautogui as p, time as t\np.click(405,75)\nt.sleep(0.8)",
            "from pyautogui import click as tap\nfrom time import sleep as wait\ntap(405,75)\nwait(0.8)",
        ]
        traces = []
        for program in programs:
            calls = []
            gui, clock = ModuleType('pyautogui'), ModuleType('time')
            gui.click = lambda *args: calls.append(('click', args))
            clock.sleep = lambda *args: calls.append(('sleep', args))
            admitted = validate_pyautogui_program(program)
            self.assertEqual(admitted, program)
            self.assertEqual(calls, [])
            with patch.dict(sys.modules, {'pyautogui': gui, 'time': clock}):
                exec(admitted, {})
            traces.append(calls)
        self.assertEqual(traces[0], [('click', (405,75)), ('sleep', (0.8,))])
        self.assertEqual(traces[0], traces[1])
        self.assertEqual(traces[0], traces[2])

    def test_sleep_alias_and_gui_configuration_are_admitted(self):
        for program in (
            "import pyautogui as p\np.hotkey('shift','f11')\np.sleep(1)",
            "import pyautogui as p\np.FAILSAFE=False\nw,h=p.size()\np.click(w/2,h/2)",
            "import pyautogui as p\np.write('openpyxl subprocess exec(')",
        ):
            with self.subTest(program=program):
                self.assertEqual(validate_pyautogui_program(program), program)

    def test_non_gui_calls_and_terminal_hotkeys_remain_rejected_with_aliases(self):
        for program in (
            'import subprocess as p\np.run("ls")',
            'import pyautogui as p\np.screenshot().save("out.png")',
            "import pyautogui as p\np.hotkey('ctrl','alt','t')",
            "import pyautogui\npyautogui.hotkey('ctrl','alt','t')",
            'from pyautogui import screenshot as shot\nshot()',
            "from pyautogui import hotkey as keys\nkeys('ctrl','alt','t')",
            'from pyautogui import *\nclick(1,2)',
            'open("out.xlsx", "w")',
        ):
            with self.subTest(program=program), self.assertRaises(ValueError):
                validate_pyautogui_program(program)

    def test_aliases_cannot_be_rebound_or_used_for_conflicting_imports(self):
        for program in (
            'import pyautogui as p\np=None\np.click(1,2)',
            'import pyautogui as p\ndel p\np.click(1,2)',
            'import pyautogui as p\nimport time as p\np.sleep(1)',
            'import pyautogui as p\ndef f(p):\n p.click(1,2)',
            'import pyautogui as p\np.click=lambda:None\np.click()',
            'from pyautogui import click as tap\ntap=None\ntap(1,2)',
        ):
            with self.subTest(program=program), self.assertRaises(ValueError):
                validate_pyautogui_program(program)
