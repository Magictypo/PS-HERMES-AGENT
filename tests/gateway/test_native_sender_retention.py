"""Live admission identity lifecycle: cancellation must not retain trust."""
from contextlib import nullcontext
from types import SimpleNamespace
import asyncio
import pytest
from gateway.session import authenticated_inbound_sender
from gateway.session_authority import SessionAuthority


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['cancelled', 'failure', 'success'])
async def test_execution_releases_sender_before_settlement(monkeypatch, outcome):
    import gateway.session_finite as finite
    authority = object.__new__(SessionAuthority)
    sender = authenticated_inbound_sender('telegram', '12345', is_human=True)
    authority.native_sender_identities = {'current': sender, 'other-session': sender}
    authority.executing = {'session': 1}
    live = SimpleNamespace(event_stream=SimpleNamespace(lock=nullcontext()))
    ref = SimpleNamespace(session_id='session')

    async def execute(*_args):
        if outcome == 'cancelled':
            raise asyncio.CancelledError
        if outcome == 'failure':
            raise RuntimeError('synthetic pre-reconstruction failure')
        return 'ok'

    monkeypatch.setattr(finite, 'execute_finite_admission', execute)
    if outcome == 'cancelled':
        with pytest.raises(asyncio.CancelledError):
            await authority._execute_claimed(ref, live, {'admission_id': 'current'})
    else:
        result = await authority._execute_claimed(ref, live, {'admission_id': 'current'})
        assert result[1] == ('failed' if outcome == 'failure' else 'completed')
    assert authority.native_sender_identities == {'other-session': sender}
    assert authority.executing == {}


@pytest.mark.parametrize('reason', ['session_busy', 'runtime_draining', 'unknown_execution'])
def test_pause_clears_only_affected_session_senders(monkeypatch, reason):
    import gateway.session_authority as module
    authority = object.__new__(SessionAuthority)
    authority.db = object()
    authority.native_sender_identities = {'current': object(), 'other-session': object()}
    other = authority.native_sender_identities['other-session']
    authority.native_waiters = {'current'}
    authority.waiters = {}
    monkeypatch.setattr(module, 'list_session_admissions', lambda *_args, **_kwargs: [{'admission_id': 'current'}])
    authority._pause(SimpleNamespace(session_id='session'), reason)
    assert authority.native_sender_identities == {'other-session': other}
    assert not authority.native_waiters
