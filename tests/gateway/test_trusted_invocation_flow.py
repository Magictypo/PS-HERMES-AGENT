"""Turn handoff and agent binding for host-owned sender provenance."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from gateway.turn_context import TurnContext
from agent.turn_facade import TurnFacadeMixin
from gateway.session import authenticated_inbound_sender
from gateway.run_turn_runner import TurnRunner
from hermes_cli.plugin_invocation_context import build_tool_invocation_context


def test_turn_facade_preserves_legacy_positional_title_and_prelude():
    import inspect

    title = object()
    prelude = object()
    legacy_call = [object(), "hello", *([None] * 12), title, prelude]
    bound = inspect.signature(TurnFacadeMixin.run_conversation).bind(*legacy_call)
    bound.apply_defaults()

    assert bound.arguments["title_user_message"] is title
    assert bound.arguments["prelude"] is prelude
    assert bound.arguments["trusted_inbound_identity"] is None


def test_turn_runner_and_agent_facade_bind_and_restore_live_sender(monkeypatch):
    import agent.aux_accounting as aux_accounting
    import agent.auxiliary_client as auxiliary_client
    import agent.background_review as background_review
    import agent.conversation_loop as conversation_loop
    import agent.interrupt_scope as interrupt_scope
    import agent.portal_tags as portal_tags
    import agent.prompt_cache_scope as prompt_cache_scope
    import agent.relay_cwd as relay_cwd
    import agent.review_idle_queue as review_idle_queue
    import agent.subagent_lifecycle as subagent_lifecycle
    import agent.turn_facade_lease as turn_facade_lease
    import agent.relay_runtime as relay_runtime
    import gateway.run as gateway_run
    import tools.approval as approval
    import tools.clarify_gateway as clarify_gateway
    import agent.notification_presentation as notification_presentation
    import hermes_cli.observability.relay_shared_metrics as relay_shared_metrics

    actor = authenticated_inbound_sender("telegram", "444444", is_human=True)
    accepted_event = SimpleNamespace(authenticated_sender=actor, internal=False)
    context = TurnContext(
        source=SimpleNamespace(platform="telegram", user_id="444444", user_name="Alice", is_bot=False),
        trusted_inbound_identity=accepted_event.authenticated_sender,
        session_id="session-1",
        session_key="telegram:shared-chat",
        message="hello",
        mute_notification_reply=False,
    )

    observed = []

    class Agent(TurnFacadeMixin):
        session_id = "session-1"
        platform = "telegram"
        _user_id = "stale-agent-user"
        _gateway_session_key = "telegram:shared-chat"
        _parent_session_id = None
        _session_db = None
        _current_turn_id = "previous-turn"

        def _conversation_root_id(self):
            return "root-session"

        def _reset_activity_labels_after_turn(self):
            return None

    agent = Agent()

    def fake_agent_loop(owner, _message, _system, _history, task_id, *_args, **_kwargs):
        invocation = build_tool_invocation_context(
            getattr(owner, "_active_trusted_inbound_identity", None), owner,
            task_id=task_id, tool_call_id="provider-call-1",
        )
        observed.append((owner._active_trusted_inbound_identity, owner._current_turn_id, invocation))
        if should_raise[0]:
            raise RuntimeError("synthetic turn failure")
        return {"final_response": "ok"}

    should_raise = [False]
    monkeypatch.setattr(conversation_loop, "run_conversation", fake_agent_loop)
    monkeypatch.setattr(background_review, "cancel_background_review_for_live_turn", lambda *_: None)
    monkeypatch.setattr(turn_facade_lease, "admit_durable_turn_lease", lambda *_args, **_kwargs: SimpleNamespace(
        early_result=None, lease=None, conversation_history=[],
    ))
    monkeypatch.setattr(turn_facade_lease, "carry_unadmitted_user_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(relay_runtime, "current_profile_key", lambda: "synthetic-profile")
    monkeypatch.setattr(relay_shared_metrics, "start_task_run", lambda **_kwargs: None)
    monkeypatch.setattr(relay_shared_metrics, "finish_task_run", lambda **_kwargs: None)
    coordinator = SimpleNamespace(
        acquire_conversation=lambda **_kwargs: object(),
        begin_turn=lambda *_args, **_kwargs: SimpleNamespace(relay_enabled=False),
        finish_logical_calls=lambda *_args, **_kwargs: None,
        end_turn=lambda *_args, **_kwargs: None,
        release_conversation=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(relay_runtime, "SESSION_COORDINATOR", coordinator)
    monkeypatch.setattr(relay_cwd, "resolve_relay_scope_cwds", lambda *_args, **_kwargs: ("", ""))
    monkeypatch.setattr(portal_tags, "set_conversation_context", lambda *_args: None)
    monkeypatch.setattr(portal_tags, "reset_conversation_context", lambda *_args: None)
    monkeypatch.setattr(portal_tags, "set_affinity_scope", lambda *_args: None)
    monkeypatch.setattr(portal_tags, "reset_affinity_scope", lambda *_args: None)
    monkeypatch.setattr(prompt_cache_scope, "declared_conversation_scope_safe", lambda *_args: None)
    monkeypatch.setattr(aux_accounting, "set_accounting_context", lambda *_args: None)
    monkeypatch.setattr(aux_accounting, "reset_accounting_context", lambda *_args: None)
    monkeypatch.setattr(auxiliary_client, "scoped_runtime_main", lambda *_args: nullcontext())
    monkeypatch.setattr(subagent_lifecycle, "bind_subagent_parent", lambda *_args: nullcontext())
    monkeypatch.setattr(interrupt_scope, "track_in_interrupt_scope", lambda *_args: nullcontext())
    monkeypatch.setattr(review_idle_queue.QUEUE, "note_turn_started", lambda: None)
    monkeypatch.setattr(review_idle_queue.QUEUE, "note_turn_finished", lambda: None)
    monkeypatch.setattr(gateway_run, "_wrap_current_message_with_observed_context", lambda message, _observed: message)
    monkeypatch.setattr(approval, "register_gateway_notify", lambda *_args: None)
    monkeypatch.setattr(approval, "unregister_gateway_notify", lambda *_args: None)
    monkeypatch.setattr(clarify_gateway, "clear_session", lambda *_args: None)
    monkeypatch.setattr(notification_presentation, "notification_turn", lambda *_args, **_kwargs: nullcontext())

    runner = object.__new__(TurnRunner)
    runner._runner = SimpleNamespace()
    runner._ctx = context
    runner._approval_owner = None
    runner._native_image_run_message = lambda: "hello"
    runner._approval_notify_sync = lambda *_args, **_kwargs: None

    assert runner._run_conversation_with_approval(agent, [], None, None, None) == {"final_response": "ok"}
    observed_sender, active_turn_id, invocation = observed[-1]
    assert observed_sender is actor
    assert active_turn_id.startswith("session-1:session-1:")
    assert invocation is not None
    assert invocation.user_id == "444444"
    assert agent._current_turn_id == "previous-turn"
    assert not hasattr(agent, "_active_trusted_inbound_identity")

    should_raise[0] = True
    with pytest.raises(RuntimeError, match="synthetic turn failure"):
        runner._run_conversation_with_approval(agent, [], None, None, None)
    assert agent._current_turn_id == "previous-turn"
    assert not hasattr(agent, "_active_trusted_inbound_identity")
