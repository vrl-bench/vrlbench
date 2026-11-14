"""A bounded SDK view that never summarizes away the live environment state."""
from pydantic import Field
from openhands.sdk.context.condenser import CondenserBase
from openhands.sdk.context.view import View
from openhands.sdk.event import MessageEvent, ObservationEvent
from openhands.sdk.llm import ImageContent, TextContent


class ObservationWindowCondenser(CondenserBase):
    recent_observations: int = Field(default=2, ge=1)
    history_chars: int = Field(default=1000, ge=128)
    image_history: int = Field(default=2, ge=1, le=40)

    def condense(self, view, agent_llm=None):
        events = list(view.events)
        observations = [i for i, e in enumerate(events) if isinstance(e, ObservationEvent)]
        real = [i for i in observations if not events[i].observation.is_error]
        def content(e):
            if isinstance(e, MessageEvent):
                return e.llm_message.content
            if isinstance(e, ObservationEvent):
                return e.observation.content
            return ()
        image_events = [i for i, e in enumerate(events)
                        if any(isinstance(p, ImageContent) for p in content(e))]
        keep_images = set(image_events[-self.image_history:])
        keep_full = set(observations[-self.recent_observations:]) | keep_images | set(real[-1:])
        projected = []
        for i, event in enumerate(events):
            if not isinstance(event, (MessageEvent, ObservationEvent)):
                projected.append(event)
                continue
            parts = []
            for part in content(event):
                if isinstance(part, ImageContent) and i not in keep_images:
                    continue
                if isinstance(event, ObservationEvent) and i not in keep_full and isinstance(part, TextContent):
                    marker = '[Historical observation, superseded; intermediate text omitted.]\n'
                    text = part.text
                    room = self.history_chars - len(marker)
                    if len(text) > room:
                        head = room // 2
                        text = text[:head] + '\n[...]\n' + text[-(room-head-7):]
                    part = part.model_copy(update={'text': marker + text})
                parts.append(part)
            if not parts:
                parts = [TextContent(text='[Historical screenshot omitted.]')]
            if isinstance(event, ObservationEvent):
                replacement = event.observation.model_copy(update={'content': parts})
                event = event.model_copy(update={'observation': replacement})
            else:
                replacement = event.llm_message.model_copy(update={'content': parts})
                event = event.model_copy(update={'llm_message': replacement})
            projected.append(event)
        return View(events=projected)


def preserve_latest_context(*, llm, recent_observations=2, history_chars=1000, image_history=2):
    """Keep task/action history and exact latest state; bound older observations.

    No model-generated summary or extra provider request is needed. ``llm`` is
    supplied by the actor's factory API and intentionally unused.
    """
    return ObservationWindowCondenser(recent_observations=recent_observations,
                                    history_chars=history_chars, image_history=image_history)
