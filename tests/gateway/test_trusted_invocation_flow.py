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


@pytest.mark.asyncio
async def test_run_agent_wrapper_passes_trusted_identity_to_inner_signature(monkeypatch):
    import inspect
    from gateway.run import GatewayRunner

    identity = authenticated_inbound_sender("telegram", "555555", is_human=True)
    runner = object.__new__(GatewayRunner)
    monkeypatch.setattr(runner, "_profile_scope_for_source", lambda _source: nullcontext())
    received = {}

    async def invoke_real_inner_signature(message, context_prompt, history, source, session_id, **kwargs):
        bound = inspect.signature(GatewayRunner._run_agent_inner).bind(
            runner, message, context_prompt, history, source, session_id, **kwargs,
        )
        received.update(bound.arguments)
        return {"final_response": "ok"}

    monkeypatch.setattr(runner, "_run_agent_inner", invoke_real_inner_signature)
    result = await runner._run_agent(
        "hello", "", [], SimpleNamespace(), "session-1",
        trusted_inbound_identity=identity,
    )

    assert result["final_response"] == "ok"
    assert received["trusted_inbound_identity"] is identity


@pytest.mark.parametrize("identity_kind", ["missing", "forged", "mismatch", "internal"])
def test_durable_native_queue_restores_only_live_branded_sender(identity_kind):
    """Native queue serialization strips trust; only the same-process admitted FIFO row may restore it."""
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from gateway.session_ingress import restore_live_native_sender

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", user_id="12345")
    event = MessageEvent(text="list products", source=source)
    if identity_kind == "missing":
        sender = None
    elif identity_kind == "forged":
        sender = SimpleNamespace(platform="telegram", user_id="12345")
    elif identity_kind == "mismatch":
        sender = authenticated_inbound_sender("telegram", "99999", is_human=True)
    else:
        sender = authenticated_inbound_sender("telegram", "12345", is_human=True)
        event.internal = True

    authority = SimpleNamespace(native_sender_identities={"admission-1": sender})
    restored = restore_live_native_sender(authority, event, "admission-1")
    assert restored is event
    assert getattr(restored, "authenticated_sender", None) is None
    assert authority.native_sender_identities == {}


def test_committed_native_admission_keeps_exact_live_identity_out_of_durable_payload():
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from gateway.session_authority import SessionAuthority
    from gateway.session_ingress import restore_live_native_sender

    authority = object.__new__(SessionAuthority)
    authority.native_sender_identities = {}
    authority._publish_pending = lambda _ref: None
    authority._receipt = lambda row: row
    sender_a = authenticated_inbound_sender("telegram", "111", is_human=True)
    sender_b = authenticated_inbound_sender("telegram", "222", is_human=True)
    source_a = SessionSource(platform=Platform.TELEGRAM, chat_id="shared", user_id="111")
    source_b = SessionSource(platform=Platform.TELEGRAM, chat_id="shared", user_id="222")
    event_a = MessageEvent(text="a", source=source_a, authenticated_sender=sender_a)
    event_b = MessageEvent(text="b", source=source_b, authenticated_sender=sender_b)

    SessionAuthority._admitted(authority, "ref", event_a, {"admission_id": "admission-a", "status": "queued"})
    SessionAuthority._admitted(authority, "ref", event_b, {"admission_id": "admission-b", "status": "queued"})
    assert set(authority.native_sender_identities) == {"admission-a", "admission-b"}

    # These events represent source reconstructed from durable native envelopes, whose
    # authenticated_sender field is intentionally absent. The process-local admission IDs
    # reattach only their own brands and consume both slots exactly once.
    restored_a = MessageEvent(text="a", source=source_a)
    restored_b = MessageEvent(text="b", source=source_b)
    assert restore_live_native_sender(authority, restored_b, "admission-b").authenticated_sender is sender_b
    assert restore_live_native_sender(authority, restored_a, "admission-a").authenticated_sender is sender_a
    assert authority.native_sender_identities == {}
    assert restore_live_native_sender(authority, MessageEvent(text="replay", source=source_a), "admission-a").authenticated_sender is None
