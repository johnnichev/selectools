"""Pins the removal of ``AgentConfig.hooks`` (deprecated v0.16, removed in v1.0).

The hooks dict was replaced by :class:`~selectools.observer.AgentObserver` /
``AsyncAgentObserver`` (see docs/decisions/002-observer-replaces-hooks.md).
Passing ``hooks=`` must now fail loudly instead of being silently accepted.
"""

from __future__ import annotations

import pytest

from selectools import AgentConfig


def test_agent_config_rejects_hooks_kwarg() -> None:
    """AgentConfig(hooks=...) must raise TypeError now that the field is gone."""
    with pytest.raises(TypeError, match="hooks"):
        AgentConfig(hooks={"on_agent_start": lambda msgs: None})  # type: ignore[call-arg]


def test_agent_config_has_no_hooks_attribute() -> None:
    config = AgentConfig()
    assert not hasattr(config, "hooks")


def test_hooks_adapter_is_gone() -> None:
    import selectools.observer as observer_module

    assert not hasattr(observer_module, "_HooksAdapter")
