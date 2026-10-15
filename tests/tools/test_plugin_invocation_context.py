"""End-to-end trusted tool context through Hermes' registry/plugin dispatch path."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from gateway.session import authenticated_inbound_sender
from hermes_cli.plugin_invocation_context import (
    bind_agent_trusted_identity,
    build_tool_invocation_context,
    current_tool_invocation_context,
)
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

_TOOL_NAME = "test_invocation_context_plugin_tool"
_META_KEY = "hermes.dev/tool-invocation-context-v1"


def _agent(sender, *, turn="turn-1", user_id="stale-agent-user"):
    return SimpleNamespace(
        session_id="shared-session",
        platform="telegram",
        _user_id=user_id,
        _gateway_session_key="telegram:shared-chat",
        _parent_session_id=None,
        _current_turn_id=turn,
        _current_api_request_id="",
        _active_trusted_inbound_identity=sender,
        _context_engine_tool_names=set(),
        _memory_manager=None,
        quiet_mode=False,
        valid_tool_names=[_TOOL_NAME],
        enabled_toolsets=None,
        disabled_toolsets=None,
    )


def _dispatch(agent, arguments, *, call_id="call-1"):
    from agent.tool_executor import _ToolCallRef, _resolve_sequential_dispatch

    ref = _ToolCallRef(_TOOL_NAME, arguments, "task-1", call_id, [])
    resolved = _resolve_sequential_dispatch(agent, ref, messages=[])
    return resolved.execute(arguments)


def test_actual_plugin_registry_path_keeps_concurrent_senders_isolated_same_session(
    tmp_path, monkeypatch
):
    """Two model tool calls share a session id but never exchange their context or MCP _meta."""
    from hermes_constants import hermes_home_key
    from tools import mcp_tool_handlers
    from tools.registry import registry

    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    barrier = threading.Barrier(2)
    seen = []
    seen_lock = threading.Lock()
    mcp_calls = []
    mcp_lock = threading.Lock()
    context = None

    def plugin_handler(arguments, **_kwargs):
        invocation = context.get_invocation_context()
        assert invocation is not None
        barrier.wait(timeout=5)
        result = context.call_mcp(
            "prima", "search_products", arguments,
            metadata=invocation.to_mcp_metadata(),
        )
        with seen_lock:
            seen.append((invocation.user_id, invocation.session_id, result["result"]))
        return json.dumps({"sender": invocation.user_id, "mcp": result})

    manager = PluginManager(scope_key=str(home))
    context = PluginContext(
        PluginManifest(name="invocation-context-test", key="invocation-context-test"), manager
    )
    registration = context.register_tool(
        name=_TOOL_NAME,
        toolset="test",
        schema={"name": _TOOL_NAME, "description": "test", "parameters": {"type": "object", "properties": {}}},
        handler=plugin_handler,
    )
    assert registration is not None
    monkeypatch.setattr(
        PluginContext, "_mcp_allowlist", staticmethod(lambda _plugin_id: ["prima"])
    )

    def fake_make_handler(server, tool, timeout, **options):
        assert (server, tool, timeout) == ("prima", "search_products", 30.0)

        def call(arguments):
            metadata = options["metadata"]
            assert set(arguments) == {"query", "user_id"}
            with mcp_lock:
                mcp_calls.append((dict(arguments), metadata))
            return json.dumps({"result": metadata[_META_KEY]["user_id"]})

        return call

    monkeypatch.setattr(mcp_tool_handlers, "_make_tool_handler", fake_make_handler)
    assert registry.get_entry(_TOOL_NAME, scope=hermes_home_key(str(home))) is not None
    senders = [
        authenticated_inbound_sender("telegram", "111111", is_human=True),
        authenticated_inbound_sender("telegram", "222222", is_human=True),
    ]

    def run(index):
        agent = _agent(senders[index], turn=f"turn-{index}")
        return json.loads(_dispatch(
            agent,
            {"query": f"q-{index}", "user_id": "model-forged"},
            call_id=f"call-{index}",
        ))

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            outputs = list(pool.map(run, range(2)))
        assert sorted((item["sender"], item["mcp"]["result"]) for item in outputs) == [
            ("111111", "111111"), ("222222", "222222"),
        ]
        assert sorted((identity, session, result) for identity, session, result in seen) == [
            ("111111", "shared-session", "111111"),
            ("222222", "shared-session", "222222"),
        ]
        assert sorted(meta[_META_KEY]["user_id"] for _args, meta in mcp_calls) == ["111111", "222222"]
        assert all("user_id" in args for args, _meta in mcp_calls)
        assert current_tool_invocation_context() is None
    finally:
        registration.dispose()


def test_missing_or_forged_identity_denies_and_exception_cleans_context(tmp_path, monkeypatch):
    from hermes_constants import hermes_home_key
    from tools import mcp_tool_handlers
    from tools.registry import registry

    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_SESSION_USER_ID", "999999")
    seen = []
    context = None

    def plugin_handler(arguments, **_kwargs):
        invocation = context.get_invocation_context()
        seen.append(invocation)
        if invocation is None:
            return json.dumps({"error": "missing trusted context"})
        if arguments.get("raise"):
            raise RuntimeError("synthetic handler failure")
        return json.dumps({"sender": invocation.user_id})

    manager = PluginManager(scope_key=str(home))
    context = PluginContext(
        PluginManifest(name="invocation-context-negative-test", key="invocation-context-negative-test"),
        manager,
    )
    registration = context.register_tool(
        name=_TOOL_NAME,
        toolset="test",
        schema={"name": _TOOL_NAME, "description": "test", "parameters": {"type": "object", "properties": {}}},
        handler=plugin_handler,
    )
    assert registration is not None
    monkeypatch.setattr(PluginContext, "_mcp_allowlist", staticmethod(lambda _plugin_id: ["prima"]))
    transport_calls = []

    def forbidden_transport(*_args, **_kwargs):
        transport_calls.append(True)
        raise AssertionError("missing/failed invocation must not reach MCP")

    monkeypatch.setattr(mcp_tool_handlers, "_make_tool_handler", forbidden_transport)
    assert registry.get_entry(_TOOL_NAME, scope=hermes_home_key(str(home))) is not None

    try:
        forged = SimpleNamespace(platform="telegram", user_id="999999")
        agent = _agent(None, user_id="999999")
        assert build_tool_invocation_context(forged, agent, task_id="task-1", tool_call_id="call") is None
        no_context_result = json.loads(_dispatch(
            agent, {"user_id": "999999", "query": "forged"}, call_id="no-context"
        ))
        assert no_context_result["error"] == "missing trusted context"
        assert seen == [None]
        assert transport_calls == []

        branded = authenticated_inbound_sender("telegram", "123456", is_human=True)
        failure = _dispatch(_agent(branded), {"raise": True}, call_id="raise")
        assert "synthetic handler failure" in failure
        assert seen[-1].user_id == "123456"
        assert current_tool_invocation_context() is None
    finally:
        registration.dispose()


def test_delegated_child_inherits_only_a_branded_parent_sender(monkeypatch):
    import run_agent
    import tools.delegate_tool as delegate
    from hermes_cli.plugin_invocation_context import ToolInvocationContext

    class Agent:
        def __init__(self, **kwargs):
            self.session_id = "child-session"
            self.platform = kwargs.get("platform", "subagent")
            self._user_id = "model-forged"
            self._gateway_session_key = ""
            self._parent_session_id = "parent-session"
            self._session_init_model_config = None
            self._print_fn = None
            self._delegate_depth = 0

    class Parent:
        pass

    parent = Parent()
    parent._delegate_depth = 0
    parent._subagent_id = None
    parent.session_id = "parent-session"
    parent._user_id = "stale-parent-user"
    parent._active_trusted_inbound_identity = authenticated_inbound_sender(
        "telegram", "333333", is_human=True
    )

    monkeypatch.setattr(run_agent, "AIAgent", Agent)
    monkeypatch.setattr(delegate, "_get_max_spawn_depth", lambda: 2)
    monkeypatch.setattr(delegate, "_get_orchestrator_enabled", lambda: False)
    monkeypatch.setattr(delegate, "_load_config", lambda: {})
    monkeypatch.setattr(delegate, "_resolve_child_toolsets", lambda *_args: (None, None))
    monkeypatch.setattr(delegate, "_build_child_system_prompt", lambda *_args, **_kwargs: "child prompt")
    monkeypatch.setattr(delegate, "_resolve_workspace_hint", lambda *_args: "")
    monkeypatch.setattr(delegate, "_resolve_child_runtime", lambda *_args, **_kwargs: {
        "provider": "synthetic", "base_url": "https://synthetic.invalid",
    })
    monkeypatch.setattr(delegate, "_build_child_progress_callback", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(delegate, "_open_child_session_db", lambda *_args: None)
    monkeypatch.setattr(delegate, "_apply_child_cache_ttl", lambda *_args: None)
    monkeypatch.setattr(delegate, "_apply_child_compression_cap", lambda *_args: None)
    monkeypatch.setattr(delegate, "_resolve_child_credential_pool", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(delegate, "_attach_child", lambda *_args: None)
    monkeypatch.setattr(delegate, "_safe_progress", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda *_args, **_kwargs: None)

    child = delegate._build_child_agent(
        0, "synthetic goal", None, None, None, 10, 1, parent
    )
    assert child._inherited_trusted_inbound_identity is parent._active_trusted_inbound_identity
    child._current_turn_id = "child-turn"
    with bind_agent_trusted_identity(child, None):
        inherited = build_tool_invocation_context(
            child._active_trusted_inbound_identity, child,
            task_id="child-task", tool_call_id="child-call",
        )
    assert isinstance(inherited, ToolInvocationContext)
    assert inherited.user_id == "333333"

    parent._active_trusted_inbound_identity = {"platform": "telegram", "user_id": "999999"}
    child_without_provenance = delegate._build_child_agent(
        0, "synthetic goal", None, None, None, 10, 1, parent
    )
    assert child_without_provenance._inherited_trusted_inbound_identity is None
    child_without_provenance._current_turn_id = "child-turn"
    with bind_agent_trusted_identity(child_without_provenance, None):
        assert build_tool_invocation_context(
            child_without_provenance._active_trusted_inbound_identity,
            child_without_provenance,
            task_id="child-task", tool_call_id="child-call",
        ) is None
