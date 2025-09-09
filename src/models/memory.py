"""Dialog memory management for maintaining conversation context."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DialogTurn:
    """Represents a single turn in the conversation."""
    query: str
    answer: str
    retrieved_docs: List[str]
    citations: Optional[List[str]] = None


class DialogMemory:
    """Manages conversation history with configurable memory size."""
    
    def __init__(self, max_turns: int = 10):
        self.turns: List[DialogTurn] = []
        self.max_turns = max_turns
    
    def add_turn(self, query: str, answer: str, retrieved_docs: List[str], 
                 citations: Optional[List[str]] = None):
        """Add a new conversation turn to memory."""
        turn = DialogTurn(query, answer, retrieved_docs, citations)
        self.turns.append(turn)
        
        # Keep only the last max_turns
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]
    
    def get_context(self, last_n_turns: int = 3) -> str:
        """Get formatted context from recent conversation turns."""
        if not self.turns:
            return ""
        
        recent_turns = self.turns[-last_n_turns:] if len(self.turns) >= last_n_turns else self.turns
        
        context = "Previous conversation context:\n"
        for i, turn in enumerate(recent_turns, 1):
            context += f"Turn {i}:\n"
            context += f"Q: {turn.query}\n"
            context += f"A: {turn.answer}\n\n"
        
        return context
    
    def has_history(self) -> bool:
        """Check if there is any conversation history."""
        return len(self.turns) > 0