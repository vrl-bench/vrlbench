"""Screenshot-only native MiniWoB controls; no DOM selectors reach the actor."""
import json

WIDTH, HEIGHT = 640, 480
RULES = ('Operate only from the current screenshot. Use one tool call per response. '
         'click uses absolute x/y pixels; type writes text to the focused control, so click the input first. '
         'stop ends an unsuccessful attempt; it never awards success. press uses a key (enter, backspace, arrowleft/right/up/down, space). '
         'drag uses x/y and to_x/to_y. Inspect the next screenshot after every action. '
         'Do not use XPath, HTML, JavaScript, or DOM inspection.')


def parse(raw):
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('GUI action must be a JSON object')
    fields = {'click': {'action','x','y'}, 'type': {'action','text'},
              'press': {'action','key'}, 'stop': {'action'}, 'drag': {'action','x','y','to_x','to_y'}}
    kind = value.get('action')
    if kind not in fields or set(value) != fields[kind]:
        raise ValueError('Use the fields for exactly one click, type, press or drag action')
    for x,y in [('x','y'),('to_x','to_y')]:
        if x in value and (type(value[x]) is not int or type(value[y]) is not int
                          or not 0 <= value[x] < WIDTH or not 0 <= value[y] < HEIGHT):
            raise ValueError('Coordinates must be integers within the 640 x 480 screenshot')
    if kind == 'type' and (not isinstance(value['text'],str) or len(value['text']) > 10000):
        raise ValueError('type requires text of at most 10000 characters')
    if kind == 'press' and value['key'] not in {'enter','backspace','arrowleft','arrowright','arrowup','arrowdown','space'}:
        raise ValueError('Unsupported GUI key')
    return value


class PointerAction:
    def __init__(self, value):
        self.value = value

    def __call__(self, driver):
        a = self.value
        def mouse(kind,x,y,buttons=0):
            driver.execute_cdp_cmd('Input.dispatchMouseEvent', {'type':kind,'x':x,'y':y,
                'button':'left' if kind != 'mouseMoved' else 'none',
                'buttons':buttons,'clickCount':1 if kind != 'mouseMoved' else 0})
        mouse('mouseMoved',a['x'],a['y'])
        mouse('mousePressed',a['x'],a['y'],1)
        if a['action']=='drag':
            for i in range(1,11):
                mouse('mouseMoved',a['x']+(a['to_x']-a['x'])*i/10,
                      a['y']+(a['to_y']-a['y'])*i/10,1)
        mouse('mouseReleased',a.get('to_x',a['x']),a.get('to_y',a['y']))

    def __str__(self):
        return json.dumps(self.value,sort_keys=True)
