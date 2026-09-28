from typing import List, Dict, Any
from app.schemas import SessionState

class MemoryManager:
    """Manages merging and deduplicating important elder memories."""

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
            value = str(item.get("value", "")).strip()
            if not value:
                continue

            category = str(item.get("category", "general")).strip().lower()
            importance = str(item.get("importance", "medium")).strip().lower()

            # Check for existing duplicate memory
            is_duplicate = False
            for existing in session.importantMemory:
                existing_val = str(existing.get("value", "")).strip().lower()
                existing_cat = str(existing.get("category", "")).strip().lower()
                if existing_val == value.lower() and existing_cat == category:
                    # Update importance if higher
                    if importance == "high":
                        existing["importance"] = "high"
                    is_duplicate = True
                    break

            if not is_duplicate:
                session.importantMemory.append({
                    "category": category,
                    "value": value,
                    "importance": importance
                })

        return session.importantMemory
