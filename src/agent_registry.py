from typing import Dict, Type, Any
from .agents.base import BaseAgent


def _make_hashable(value: Any):
    if isinstance(value, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple, set)):
        return tuple(_make_hashable(v) for v in value)
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


class AgentRegistry:
    """Registry for managing multiple agents in the system."""

    def __init__(self):
        self._agents: Dict[str, Type[BaseAgent]] = {}
        self._instances: Dict[str, BaseAgent] = {}

    def register(self, name: str, agent_class: Type[BaseAgent]) -> None:
        """
        Register an agent class with a given name.

        Args:
            name: Unique identifier for the agent
            agent_class: The agent class to register
        """
        if name in self._agents:
            raise ValueError(f"Agent '{name}' is already registered")
        self._agents[name] = agent_class

    def get_agent(self, name: str, **kwargs) -> BaseAgent:
        """
        Get an agent instance by name. Creates a new instance if not cached.

        Args:
            name: Name of the registered agent
            **kwargs: Arguments to pass to the agent constructor

        Returns:
            Instance of the requested agent
        """
        if name not in self._agents:
            raise ValueError(f"Agent '{name}' is not registered")

        # Create cache key based on name and kwargs
        hashable_kwargs = tuple(sorted((k, _make_hashable(v)) for k, v in kwargs.items()))
        cache_key = f"{name}_{hash(hashable_kwargs)}"

        if cache_key not in self._instances:
            self._instances[cache_key] = self._agents[name](**kwargs)

        return self._instances[cache_key]

    def list_agents(self) -> list[str]:
        """Return list of all registered agent names."""
        return list(self._agents.keys())

    def unregister(self, name: str) -> None:
        """
        Unregister an agent.

        Args:
            name: Name of the agent to unregister
        """
        if name in self._agents:
            del self._agents[name]
            # Clear cached instances for this agent
            self._instances = {
                k: v for k, v in self._instances.items()
                if not k.startswith(f"{name}_")
            }


# Global registry instance
_global_registry = AgentRegistry()


def register_agent(name: str, agent_class: Type[BaseAgent]) -> None:
    """Register an agent in the global registry."""
    _global_registry.register(name, agent_class)


def get_agent(name: str, **kwargs) -> BaseAgent:
    """Get an agent from the global registry."""
    return _global_registry.get_agent(name, **kwargs)


def list_agents() -> list[str]:
    """List all registered agents."""
    return _global_registry.list_agents()
