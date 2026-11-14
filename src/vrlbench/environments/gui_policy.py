"""Static admission for programs restricted to direct PyAutoGUI operations."""

from __future__ import annotations

import ast


_MOUSE_KEYBOARD_CALLS = frozenset({
    'click', 'doubleClick', 'moveTo', 'dragTo', 'scroll', 'write', 'typewrite',
    'press', 'hotkey', 'keyDown', 'keyUp', 'mouseDown', 'mouseUp', 'rightClick',
    'middleClick', 'tripleClick', 'move', 'moveRel', 'drag', 'dragRel', 'hscroll',
    'vscroll', 'sleep', 'size', 'position',
})


def validate_pyautogui_program(program: str) -> str:
    """Check direct imports/calls and return the original program unchanged.

    Import aliases retain their Python meaning. Module bindings must remain
    stable so call admission can resolve them statically. Execution belongs to
    the isolated desktop environment; this function does not execute code.
    """
    if program in {'DONE', 'FAIL', 'WAIT'}:
        return program
    tree = ast.parse(program)
    bindings = {'pyautogui': 'pyautogui', 'time': 'time'}
    functions = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            allowed = _MOUSE_KEYBOARD_CALLS if node.module == 'pyautogui' else {'sleep'}
            if node.level or node.module not in {'pyautogui', 'time'}:
                raise ValueError('GUI-only: imports limited to pyautogui/time')
            for item in node.names:
                if item.name not in allowed:
                    raise ValueError('GUI-only: imports limited to mouse/keyboard/sleep functions')
                name, target = item.asname or item.name, (node.module, item.name)
                if name in bindings or name in functions and functions[name] != target:
                    raise ValueError('GUI-only: alias has conflicting imports')
                functions[name] = target
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name not in {'pyautogui', 'time'}:
                    raise ValueError('GUI-only: imports limited to pyautogui/time')
                name = item.asname or item.name
                if name in functions or name in bindings and bindings[name] != item.name:
                    raise ValueError('GUI-only: module alias has conflicting imports')
                bindings[name] = item.name

    protected = set(bindings) | set(functions)
    for node in ast.walk(tree):
        rebound = (
            isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
            and node.id in protected
        ) or (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name in protected
        ) or (isinstance(node, ast.arg) and node.arg in protected) or (
            isinstance(node, ast.ExceptHandler) and node.name in protected
        )
        if rebound:
            raise ValueError('GUI-only: import aliases cannot be rebound')
        if (isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del))
                and isinstance(node.value, ast.Name) and node.value.id in bindings
                and node.attr in _MOUSE_KEYBOARD_CALLS):
            raise ValueError('GUI-only: admitted methods cannot be replaced')
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        module, operation = None, None
        if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name):
            module, operation = bindings.get(function.value.id), function.attr
        elif isinstance(function, ast.Name):
            module, operation = functions.get(function.id, (None, None))
        allowed = (
            module == 'pyautogui' and operation in _MOUSE_KEYBOARD_CALLS
        ) or (module == 'time' and operation == 'sleep')
        if not allowed:
            raise ValueError('GUI-only: mouse/keyboard/sleep calls only')
        if module == 'pyautogui' and operation == 'hotkey':
            keys = {arg.value.lower() for arg in node.args
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
            if {'ctrl', 'alt', 't'} <= keys:
                raise ValueError('GUI-only: terminal launch is excluded')
    return program
