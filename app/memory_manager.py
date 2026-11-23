from typing import List, Dict, Any
from app.schemas import SessionState
from app.state.reducer import StateReducer

class MemoryManager:
    """Manages merging and deduplicating important elder memories via StateReducer."""

    @staticmethod
    def add_memories(session: SessionState, new_memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge memory updates while preventing duplicate items.
        Deduplicates based on category and value similarity.
        """
        if not new_memories or not isinstance(new_memories, list):
            return session.importantMemory

        for item in new_memories:
            if not isinstance(item, dict):
                continue
            val = item.get("value")
            if val:
                StateReducer._merge_memory(
                    session,
                    category=item.get("category", "general"),
                    value=val,
                    importance=item.get("importance", "medium")
                )

        return session.importantMemory
