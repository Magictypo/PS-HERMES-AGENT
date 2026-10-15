"""Immutable, per-tool context from an authenticated inbound gateway event.

This is a host-to-plugin context carrier, not an authentication mechanism for a
remote service. Plugins must use a protected transport/backend verifier before
authorizing an operation. There is deliberately no environment or model-argument
fallback.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True, slots=True)
class ToolInvocationContext:
    """Read-only correlation and sender data for one actual tool invocation."""

    platform: str
    user_id: str
    session_id: str
    turn_id: str
    task_id: str
    tool_call_id: str
    session_key: str
    parent_session_id: str

    def to_mcp_metadata(self) -> dict[str, Any]:
        """Return a fresh, out-of-band MCP ``_meta`` carrier for this context.

        The metadata is a transport claim, not a signature. The receiving server
        must authenticate the client transport and resolve/authorize the sender
        on its own backend.
        """
        return {
            "hermes.dev/tool-invocation-context-v1": {
                "platform": self.platform,
                "user_id": self.user_id,
                "session_id": self.session_id,
                "turn_id": self.turn_id,
                "task_id": self.task_id,
                "tool_call_id": self.tool_call_id,
                "session_key": self.session_key,
            }
        }


_CURRENT_TOOL_INVOCATION: ContextVar[ToolInvocationContext | None] = ContextVar(
    "hermes_plugin_tool_invocation_context", default=None
)


def current_tool_invocation_context() -> ToolInvocationContext | None:
    """Return only the context explicitly bound by Hermes for this tool call."""
    return _CURRENT_TOOL_INVOCATION.get()


_AGENT_IDENTITY_MISSING = object()


@contextmanager
def bind_agent_trusted_identity(agent: Any, requested: Any) -> Iterator[None]:
    """Bind the accepted sender for one leased agent turn, restoring it on every exit."""
    from gateway.session import is_authenticated_inbound_sender

    previous = getattr(agent, "_active_trusted_inbound_identity", _AGENT_IDENTITY_MISSING)
    inherited = getattr(agent, "_inherited_trusted_inbound_identity", None)
    identity = requested if is_authenticated_inbound_sender(requested) else inherited
    if not is_authenticated_inbound_sender(identity):
        identity = None
    agent._active_trusted_inbound_identity = identity
    try:
        yield
    finally:
        if previous is _AGENT_IDENTITY_MISSING:
            try:
                delattr(agent, "_active_trusted_inbound_identity")
            except AttributeError:
                pass
        else:
            agent._active_trusted_inbound_identity = previous


@contextmanager
def bind_tool_invocation_context(context: ToolInvocationContext | None) -> Iterator[None]:
    """Bind one invocation to a handler and always restore the previous context."""
    token: Token = _CURRENT_TOOL_INVOCATION.set(context)
    try:
        yield
    finally:
        _CURRENT_TOOL_INVOCATION.reset(token)


def _required_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value or value.strip() != value or len(value) > 512:
        return None
    return value


def build_tool_invocation_context(
    authenticated_sender: Any,
    agent: Any,
    *,
    task_id: Any,
    tool_call_id: Any,
) -> ToolInvocationContext | None:
    """Create a context only from a branded gateway sender and matching live turn.

    The provenance object is minted by a platform adapter from its native inbound
    human event. A restored ``SessionSource.user_id``, ``AIAgent._user_id``, an env
    var, or a value in tool arguments is insufficient.
    """
    try:
        from gateway.session import is_authenticated_inbound_sender

        if not is_authenticated_inbound_sender(authenticated_sender):
            return None
        platform = _required_text(authenticated_sender.platform)
        user_id = _required_text(authenticated_sender.user_id)
        session_id = _required_text(getattr(agent, "session_id", None))
        turn_id = _required_text(getattr(agent, "_current_turn_id", None))
        task = _required_text(task_id)
        call_id = _required_text(tool_call_id)
        agent_platform = _required_text(getattr(agent, "platform", None))
        if not all((platform, user_id, session_id, turn_id, task, call_id, agent_platform)):
            return None
        if agent_platform != "subagent" and agent_platform != platform:
            return None
        return ToolInvocationContext(
            platform=platform,
            user_id=user_id,
            session_id=session_id,
            turn_id=turn_id,
            task_id=task,
            tool_call_id=call_id,
            session_key=str(getattr(agent, "_gateway_session_key", None) or ""),
            parent_session_id=str(getattr(agent, "_parent_session_id", None) or ""),
        )
    except Exception:
        # Malformed or ambiguous provenance is never repaired from ambient state.
        return None
