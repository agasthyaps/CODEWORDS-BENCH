"""Agent state management with scratchpad support."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """Per-agent persistent state within a game.
    
    The scratchpad is append-only and private to each agent.
    It persists across turns within a single game episode.
    """
    agent_id: str
    scratchpad: str = ""
    
    def append_to_scratchpad(self, turn: int, content: str) -> None:
        """Append new content to scratchpad with turn marker.
        
        Args:
            turn: The current turn number
            content: Content to append (stripped of whitespace)
        """
        content = content.strip()
        if content:
            if self.scratchpad:
                self.scratchpad += f"\n[Turn {turn}] {content}"
            else:
                self.scratchpad = f"[Turn {turn}] {content}"


class AgentStateManager:
    """Manages agent states for a game session.
    
    Creates and tracks AgentState instances for each agent in a game.
    States persist across turns but are reset between episodes.
    """
    
    def __init__(self):
        self._states: dict[str, AgentState] = {}
    
    def get_or_create(self, agent_id: str) -> AgentState:
        """Get existing state or create new one for agent.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            AgentState for the specified agent
        """
        if agent_id not in self._states:
            self._states[agent_id] = AgentState(agent_id=agent_id)
        return self._states[agent_id]
    
    def get_scratchpad(self, agent_id: str) -> str:
        """Get scratchpad content for an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            Scratchpad content, or empty string if agent has no state
        """
        state = self._states.get(agent_id)
        return state.scratchpad if state else ""
    
    def get_all_states(self) -> dict[str, AgentState]:
        """Get all agent states.
        
        Returns:
            Dictionary mapping agent_id to AgentState
        """
        return dict(self._states)
    
    def reset(self) -> None:
        """Reset all agent states (for new episode)."""
        self._states.clear()
