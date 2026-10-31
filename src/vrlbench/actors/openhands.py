"""OpenHands' native agent loop over VRL-Bench's controlled environment tools.

The SDK owns the within-trial conversation. VRL-Bench still owns environment
resets, action admission, verified outcomes, and memory between fresh trials.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, ClassVar, Mapping
from uuid import uuid4

from vrlbench.environments.interactive import InteractiveActor, Observation, Session

if sys.version_info < (3, 12):
    raise RuntimeError("The OpenHands actor requires Python 3.12 or newer")

try:
    from pydantic import Field, PrivateAttr
    from openhands.sdk import Agent, LLM
    from openhands.sdk.context import AgentContext
    from openhands.sdk.context.condenser import CondenserBase
    from openhands.sdk.conversation import LocalConversation
    from openhands.sdk.event import AgentErrorEvent, ObservationEvent
    from openhands.sdk.llm import ImageContent, Message, TextContent
    from openhands.sdk.tool import Action, Observation as SDKObservation
    from openhands.sdk.tool import Tool, ToolDefinition, ToolExecutor, register_tool
except ImportError as exc:
    raise ImportError("Install vrlbench[openhands] to use the OpenHands actor") from exc


class EnvironmentAction(Action):
    action: str = Field(description="Exactly one action in the environment's native format")


class EnvironmentObservation(SDKObservation):
    pass


_TRIAL_ACTORS: dict[str, "OpenHandsActor"] = {}


class EnvironmentExecutor(ToolExecutor):
    def __init__(self, actor: "OpenHandsActor") -> None:
        self.actor = actor

    def __call__(self, action: EnvironmentAction, conversation: Any = None) -> EnvironmentObservation:
        from vrlbench.runtime import llm as runtime_llm

        # SDK tools may execute on a worker thread. Keep native evaluator
        # model calls on this same case/trial usage ledger.
        with runtime_llm.token_budget_scope(self.actor._token_ledger):
            return self.actor.execute_environment_action(action.action)


class EnvironmentActionTool(ToolDefinition):
    name: ClassVar[str] = "environment_action"

    @classmethod
    def create(cls, conv_state: Any, **_: Any) -> list["EnvironmentActionTool"]:
        actor = _TRIAL_ACTORS[str(conv_state.id)]
        return [cls(description=("Execute one native action in the current task environment. "
                                 "The action argument must be a complete native statement, "
                                 "preserving prefixes and quoted string arguments.\n" +
                                 actor.session.system_prompt),
                    action_type=EnvironmentAction, observation_type=EnvironmentObservation,
                    executor=EnvironmentExecutor(actor))]


register_tool("VRLBenchEnvironmentAction", EnvironmentActionTool)


class AccountedLLM(LLM):
    """Keep SDK generation native, with admission and usage on every SDK call."""

    _actor: Any = PrivateAttr(default=None)

    def generate(self, messages: list[Any], tools: Any = None, **kwargs: Any) -> Any:
        try:
            request = self._actor.before_sdk_request(self, messages, tools)
        except BaseException as exc:
            self._actor._fatal_error = exc
            raise
        try:
            response = super().generate(messages=messages, tools=tools, **kwargs)
        except BaseException as exc:
            self._actor._fatal_error = exc
            self._actor.request_observer.on_error(request, exc)
            raise
        try:
            self._actor.after_sdk_response(request, response, tools)
        except BaseException as exc:
            self._actor._fatal_error = exc
            raise
        return response

    async def agenerate(self, messages: list[Any], tools: Any = None, **kwargs: Any) -> Any:
        try:
            request = self._actor.before_sdk_request(self, messages, tools)
        except BaseException as exc:
            self._actor._fatal_error = exc
            raise
        try:
            response = await super().agenerate(messages=messages, tools=tools, **kwargs)
        except BaseException as exc:
            self._actor._fatal_error = exc
            self._actor.request_observer.on_error(request, exc)
            raise
        try:
            self._actor.after_sdk_response(request, response, tools)
        except BaseException as exc:
            self._actor._fatal_error = exc
            raise
        return response


class OpenHandsActor(InteractiveActor):
    """One fresh native SDK Conversation for each VRL-Bench trial."""

    def __init__(self, session: Session, *, model: str, max_environment_steps: int,
                 artifact_dir: str | Path, capture_observations: bool = False,
                 on_progress: Callable[[dict[str, Any]], None] | None = None,
                 actor_params: Mapping[str, Any] | None = None,
                 sdk_model_config: Mapping[str, Any] | None = None,
                 sdk_llm_factory: Callable[..., Any] | None = None,
                 sdk_request_observer: Any = None) -> None:
        params = dict(actor_params or {})
        unknown = set(params) - {"max_output_tokens", "condenser"}
        if unknown:
            raise ValueError(f"unknown OpenHands actor parameters: {sorted(unknown)}")
        self.max_output_tokens = params.get("max_output_tokens", 1024)
        if type(self.max_output_tokens) is not int or self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be a positive integer")
        self.condenser_spec = params.get("condenser")
        if self.condenser_spec is not None and not isinstance(self.condenser_spec, Mapping):
            raise TypeError("condenser must be a trusted FactorySpec")
        self.sdk_model_config = dict(sdk_model_config or {})
        self.sdk_llm_factory = sdk_llm_factory
        self.request_observer = sdk_request_observer
        if self.request_observer is None:
            from vrlbench.runtime.llm import external_request_observer

            self.request_observer = external_request_observer(model)
        for name in ("before_request", "on_response", "on_error"):
            if not callable(getattr(self.request_observer, name, None)):
                raise TypeError(f"SDK request observer requires {name}()")
        self.sdk_version = version("openhands-sdk")
        self.conversation: Any = None
        self.condenser_identity: dict[str, Any] | None = None
        super().__init__(session, model=model, max_environment_steps=max_environment_steps,
                         artifact_dir=artifact_dir, capture_observations=capture_observations,
                         on_progress=on_progress)

    def set_trial_context(self, *, trial_index: int, total_trials: int) -> dict[str, Any]:
        result = super().set_trial_context(trial_index=trial_index, total_trials=total_trials)
        self.sdk_events: list[dict[str, Any]] = []
        self.raw_event_path: Path | None = None
        self._event_digest = hashlib.sha256()
        self._event_stream: Any = None
        self._image_references: dict[str, dict[str, Any]] = {}
        self._response: dict[str, Any] | None = None
        self._request_count = 0
        self._fatal_error: BaseException | None = None
        return result

    def _make_llm(self) -> LLM:
        config = self.sdk_model_config
        if config.get("transport") is not None and self.sdk_llm_factory is None:
            raise ValueError("OpenHands requires a native SDK LLM factory for custom transports")
        model_name = str(config.get("name") or self.model)
        if config.get("provider", "openai_compatible") == "openai_compatible" and "/" not in model_name:
            model_name = "openai/" + model_name
        kwargs = {"model": model_name,
                  "api_key": os.environ.get(str(config.get("api_key_env", "OPENAI_API_KEY"))),
                  "base_url": config.get("base_url"),
                  "timeout": int(config.get("timeout_seconds", 120)),
                  "usage_id": "vrlbench_actor", "num_retries": 0,
                  "max_output_tokens": self.max_output_tokens, "stream": False}
        if config.get("provider", "openai_compatible") == "openai_compatible":
            kwargs["api_mode"] = "chat"
        options = dict(config.get("request_options") or {})
        # Protocol and output limits belong to the SDK and actor, not a transport map.
        protocol = {"model", "messages", "tools", "stream", "num_retries",
                    "max_tokens", "max_completion_tokens", "max_output_tokens",
                    "api_key", "base_url", "api_version", "usage_id",
                    "provider_connection_id", "auth_type", "subscription_vendor",
                    "fallback_strategy"}
        forbidden = protocol & set(options)
        extra_body = options.get("extra_body")
        if isinstance(extra_body, Mapping):
            forbidden |= protocol & set(extra_body)
        if forbidden:
            raise ValueError(f"OpenHands request_options cannot override {sorted(forbidden)}")
        if "n" in options:
            count = options.pop("n")
            if type(count) is not int or count != 1:
                raise ValueError("OpenHands requests support exactly one completion")
        if "extra_body" in options:
            kwargs["litellm_extra_body"] = options.pop("extra_body")
        unknown = set(options) - set(LLM.model_fields)
        if unknown:
            raise ValueError(f"OpenHands request_options are not SDK LLM fields: {sorted(unknown)}")
        kwargs.update(options)
        if self.sdk_llm_factory is None:
            llm = AccountedLLM(**kwargs)
            llm._actor = self
        else:
            # A trusted factory must use the same callbacks for every actual request.
            llm = self.sdk_llm_factory(kwargs, self)
        if not isinstance(llm, LLM):
            raise TypeError("sdk_llm_factory must return an OpenHands SDK LLM")
        return llm

    @staticmethod
    def _is_actor_request(tools: Any) -> bool:
        return any(getattr(tool, "name", None) == EnvironmentActionTool.name for tool in tools or ())

    def before_sdk_request(self, llm: LLM, messages: list[Any], tools: Any) -> Any:
        actor_request = self._is_actor_request(tools)
        if actor_request:
            self.ledger.require_step()
        request = self.request_observer.before_request(messages, llm.max_output_tokens)
        self._request_count += 1
        if actor_request:
            self.ledger.charge_actor_turn()
        return request

    def after_sdk_response(self, request: Any, response: Any, tools: Any) -> None:
        self.request_observer.on_response(request, response.raw_response)
        if not self._is_actor_request(tools):
            return
        message = response.message
        raw = message.model_dump_json(exclude_none=True)
        calls = list(message.tool_calls or [])
        self._response = {"raw": raw, "id": response.id, "executed": False,
                          "rejected": False, "feedback": ""}
        reason = None
        if len(calls) != 1:
            reason = "return exactly one environment_action tool call"
        elif calls[0].name != EnvironmentActionTool.name:
            reason = "only environment_action can act; a finish declaration is not environment success"
        else:
            try:
                arguments = json.loads(calls[0].arguments)
                if not isinstance(arguments, dict) or not isinstance(arguments.get("action"), str):
                    raise ValueError("environment_action requires an action string")
            except (ValueError, TypeError) as exc:
                reason = f"invalid environment_action arguments: {exc}"
        if reason:
            self._reject_sdk_response(reason)

    def _reject_sdk_response(self, reason: str) -> str:
        assert self._response is not None
        if not self._response["rejected"]:
            self._response["rejected"] = True
            self._response["feedback"] = self._reject_action(self._response["raw"], reason)
        return str(self._response["feedback"])

    def _sdk_content(self, observation: Observation, text: str) -> list[Any]:
        content: list[Any] = [TextContent(text=text)]
        if observation.image is not None:
            mime = "image/png" if observation.image.startswith(b"\x89PNG") else "image/jpeg"
            url = f"data:{mime};base64," + base64.b64encode(observation.image).decode("ascii")
            digest = hashlib.sha256(observation.image).hexdigest()
            reference = next(item for item in reversed(self.screenshots) if item["sha256"] == digest)
            self._image_references[url] = reference
            content.append(ImageContent(image_urls=[url]))
        return content

    def execute_environment_action(self, raw: str) -> EnvironmentObservation:
        if self._response is None:
            raise RuntimeError("an SDK environment action requires its recorded model response")
        if self._response["rejected"]:
            return EnvironmentObservation.from_text(self._response["feedback"], is_error=True)
        if self._response["executed"]:
            raise RuntimeError("one SDK model response cannot execute multiple environment actions")
        self._response["executed"] = True
        try:
            action = self._action_from_response(raw)
        except ValueError as exc:
            return EnvironmentObservation.from_text(self._reject_sdk_response(str(exc)), is_error=True)
        try:
            observation, text = self._dispatch_action(action, raw_response=self._response["raw"])
        except BaseException as exc:
            # The SDK may turn executor faults into tool observations. Retain the
            # original exception and fail the run; infrastructure faults are not scores.
            self._fatal_error = exc
            raise
        feedback = f"{text}\n\nEnvironment actions remaining: {self.ledger.remaining_steps}."
        if self.close_reason != "running":
            feedback += f"\nEnvironment trial ended: {self.close_reason}."
        return EnvironmentObservation(content=self._sdk_content(observation, feedback))

    def _on_sdk_event(self, event: Any) -> None:
        payload = event.model_dump(mode="json")
        encoded = (json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        self._event_stream.write(encoded)
        self._event_stream.flush()
        self._event_digest.update(encoded)
        self.sdk_events.append(payload)
        if isinstance(event, AgentErrorEvent) and self._response is not None:
            if not self._response["executed"] and not self._response["rejected"]:
                self._reject_sdk_response(event.error)
        rejected = bool(self._response and self._response["rejected"])
        if (rejected or self.close_reason != "running" or not self.ledger.remaining_steps
                or self._fatal_error is not None):
            # This callback runs after the SDK has recorded the event. Pause before
            # the native loop can request another model response or mutate the env.
            if self.conversation is not None:
                self.conversation.pause()
        elif isinstance(event, ObservationEvent):
            self._checkpoint("environment_feedback")

    def __call__(self) -> str:
        from vrlbench.runtime import llm as runtime_llm

        self._token_ledger = runtime_llm._get_token_ledger()
        self.system_context = runtime_llm.get_context()
        self.user_context = runtime_llm.get_user_context()
        prepared = self.prepare_trial()
        assert self.prepared is not None
        directory = self.artifact_dir / f"trial_{self.trial_index:03d}" / "openhands"
        directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.raw_event_path = directory / "events.jsonl"
        llm = self._make_llm()
        context = AgentContext(system_message_suffix=self.system_context or None,
                               current_datetime=None, load_user_skills=False,
                               load_public_skills=False, load_project_skills=False,
                               load_memory=False)
        agent_kwargs: dict[str, Any] = {
            "llm": llm, "tools": [Tool(name="VRLBenchEnvironmentAction")],
            "include_default_tools": [], "tool_concurrency_limit": 1,
            "agent_context": context,
            "system_prompt": (
                "Interact through exactly one environment_action tool call per response. "
                "Place the complete native action in that tool's action argument; "
                "do not return it as an ordinary assistant message. The native instructions "
                "below describe the argument's syntax, not the assistant-message format. "
                "Python programs must include their required imports and qualified API names. "
                "Pass native DONE, FAIL or WAIT markers through the same tool when supported. "
                "Only the environment determines task success.\n\nNative action instructions:\n"
                + self.session.system_prompt),
        }
        if self.condenser_spec is not None:
            from vrlbench.plugins import create_plugin

            kwargs = dict(self.condenser_spec.get("kwargs", {}))
            if "llm" in kwargs:
                raise ValueError("condenser factories receive the accounted SDK LLM automatically")
            condenser, self.condenser_identity = create_plugin(
                {**self.condenser_spec, "kwargs": {**kwargs, "llm": llm}})
            if not isinstance(condenser, CondenserBase):
                raise TypeError("condenser factory must return an OpenHands CondenserBase")
            agent_kwargs["condenser"] = condenser
        conversation_id = uuid4()
        _TRIAL_ACTORS[str(conversation_id)] = self
        descriptor = os.open(self.raw_event_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._event_stream = os.fdopen(descriptor, "wb")
        try:
            self.conversation = LocalConversation(
                agent=Agent(**agent_kwargs), workspace=directory / "workspace",
                conversation_id=conversation_id, callbacks=[self._on_sdk_event],
                persistence_dir=directory / "sdk", visualizer=None,
                profile_store_dir=directory / "profiles",
                max_iteration_per_run=self.ledger.max_steps_per_episode * 3 + 4,
                stuck_detection=False, delete_on_close=False)
            initial = f"Task: {prepared['task']}\nObservation:\n{prepared['initial_observation']}"
            initial += f"\nEnvironment actions remaining: {self.ledger.remaining_steps}."
            if self.user_context:
                initial = self.user_context + "\n\n" + initial
            self.conversation.send_message(Message(role="user", content=self._sdk_content(self.prepared.observation, initial)))
            # The SDK discovers only exact LLM classes; register this accounted
            # subclass explicitly so its native metrics also reach ConversationStats.
            if llm.usage_id not in self.conversation.llm_registry.list_usage_ids():
                self.conversation.llm_registry.add(llm)
            if set(self.conversation.agent.tools_map) != {EnvironmentActionTool.name}:
                raise RuntimeError("OpenHands may only expose the controlled environment_action tool")
            while self.close_reason == "running" and self.ledger.remaining_steps:
                before = self.ledger.action_attempts
                self.conversation.run()
                if self._fatal_error is not None:
                    raise self._fatal_error
                if self.close_reason != "running" or not self.ledger.remaining_steps:
                    break
                if self._response is None or self.ledger.action_attempts == before:
                    raise RuntimeError("OpenHands stopped without an accounted action response")
                # A plain assistant completion stops the SDK's run, but not the
                # environment. It has already consumed one invalid opportunity.
                feedback = str(self._response["feedback"] or
                    f"The environment has not terminated. Execute one environment_action; "
                    f"{self.ledger.remaining_steps} action opportunities remain.")
                self._response = None
                self.conversation.send_message(feedback)
            if self.close_reason == "running":
                self._finish(terminated=False, truncated=True)
        finally:
            try:
                if self.conversation is not None:
                    self.conversation.close()
            finally:
                _TRIAL_ACTORS.pop(str(conversation_id), None)
                self._event_stream.close()
                self._event_stream = None
                self.conversation = None
        outcome = "SUCCESS" if self.success else "FAIL"
        assert self.evidence is not None
        return json.dumps(self.evidence, ensure_ascii=False) + f"\nSTATUS: {outcome}\nClose reason: {self.close_reason}"

    def _metadata_snapshot(self) -> dict[str, Any]:
        return {**super()._metadata_snapshot(), "actor_name": "openhands",
                "actor_sdk_version": self.sdk_version,
                "sdk_requests": self._request_count,
                "sdk_event_log": self._raw_event_reference(),
                "actor_trajectory_format": "openhands_events_text_projection_v1",
                "condenser_identity": deepcopy(self.condenser_identity)}

    def _raw_event_reference(self) -> dict[str, Any] | None:
        if self.raw_event_path is None:
            return None
        return {"path": str(self.raw_event_path), "sha256": self._event_digest.hexdigest(),
                "events": len(self.sdk_events)}

    def get_actor_trajectory(self) -> str:
        def text_projection(value: Any) -> Any:
            if isinstance(value, str) and value in self._image_references:
                reference = self._image_references[value]
                return f"Screenshot: {reference['path']} (sha256={reference['sha256']})"
            if isinstance(value, dict):
                return {key: text_projection(item) for key, item in value.items()}
            if isinstance(value, list):
                return [text_projection(item) for item in value]
            return value

        return json.dumps({"format": "openhands_events_text_projection_v1",
                           "raw_event_log": self._raw_event_reference(),
                           "events": text_projection(self.sdk_events)},
                          ensure_ascii=False, allow_nan=False)
