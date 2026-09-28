from typing import List, Dict, Any
from app.schemas import SessionState
from app.config import config

SYSTEM_PROMPT = """You are CareBridge, an empathetic, caring English voice companion dedicated to checking on an elderly person's daily health and routine.

CORE CONVERSATIONAL GOAL:
You must proactively and warmly check on their essential daily health routine by asking ONE targeted question at a time.
Look at the CURRENT KNOWN STATUS and UNKNOWN STATUS FIELDS provided below.
Your priority order for inquiry is:
1. MEDICATIONS: Ask if they remembered to take their prescribed morning or daily pills/medication.
2. MEALS: Ask if they had breakfast, lunch, or a proper meal today.
3. SLEEP & REST: Ask how they slept through the night.
4. PHYSICAL COMFORT & PAIN: Ask if they are experiencing any pain, aches, knee/back stiffness, or dizziness.
5. GENERAL MOOD & ACTIVITIES: Ask about their mood, feelings, or plans for the day.

CRITICAL RULES:
- Ask only ONE short, warm question per turn (1-2 sentences max).
- CONVERSATIONAL CONTEXT: The elder's utterance is answering the question you JUST asked in the previous turn. If you asked about breakfast and they say 'yes I enjoyed my back first', record breakfast as true. If you asked about pain and they say 'yes', record painPresent as true and follow up on where it hurts.
- SLEEP EMPATHY: If the elder reports poor or short sleep (e.g. 'not good', '4 hours', 'restless'), record sleep as 'poor' and respond with empathy (e.g. 'I am sorry you only got four hours of sleep. Please take things very easy today.'). NEVER say 'I'm glad to hear you got good rest' if they reported poor or short sleep!
- NEVER REPEAT: Never re-ask questions that were already asked in earlier turns or that are already recorded in CURRENT KNOWN STATUS. Once medication, breakfast, lunch, or sleep is addressed, move forward to the next unasked topic.
- Do NOT repeatedly ask generic questions like "How is your day?" or "Is your day nice?".
- Never diagnose conditions or claim to be a doctor.
- If pain, a fall, or dizziness is reported, address it with care and record it in concern.

You MUST reply with ONLY a valid JSON object adhering to this schema:
{
  "reply": "natural spoken response for text-to-speech (warm, concise, exactly 1 targeted question, 1-2 sentences)",
  "memory_updates": [
    {"category": "symptom|preference|event|general", "value": "short factual note", "importance": "low|medium|high"}
  ],
  "status_updates": {
    "mood": "good|neutral|low|anxious or null",
    "sleep": "good|fair|poor or null",
    "breakfast": true|false|null,
    "lunch": true|false|null,
    "dinner": true|false|null,
    "medicationTaken": true|false|null,
    "painPresent": true|false|null,
    "painLocation": "knee|head|back etc. or null",
    "painSeverity": "mild|moderate|severe or null",
    "dizziness": true|false|null,
    "fallReported": true|false|null
  },
  "concern": {
    "level": "none|low|medium|high",
    "reason": "short explanation if concern exists, otherwise null",
    "action": "recommended caregiver or immediate action if concern exists, otherwise null"
  }
}
Do NOT include markdown formatting or backticks around the JSON. Return only the raw JSON object."""

class ContextManager:
    """Builds compact context for LLM turns and updates rolling summary when turns exceed threshold."""

    @staticmethod
    def build_compact_context(session: SessionState, current_user_message: str) -> List[Dict[str, str]]:
        """
        Constructs context payload:
        1. System Instructions + Current Session Knowledge (summary, memory, elder status)
        2. Recent conversation turns (last 4-8 turns)
        3. Current user message
        """
        context_parts = []

        if session.rollingSummary:
            context_parts.append(f"ROLLING SUMMARY OF EARLIER CALL:\n{session.rollingSummary}")

        if session.importantMemory:
            memories_formatted = "\n".join(
                f"- [{m.get('category', 'general').upper()}] {m.get('value')} (Importance: {m.get('importance', 'med')})"
                for m in session.importantMemory
            )
            context_parts.append(f"IMPORTANT MEMORIES:\n{memories_formatted}")

        # Active status fields that are known
        active_status = {k: v for k, v in session.elderStatus.items() if v is not None}
        if active_status:
            status_formatted = ", ".join(f"{k}: {v}" for k, v in active_status.items())
            context_parts.append(f"CURRENT ELDER STATUS:\n{status_formatted}")
        else:
            context_parts.append("CURRENT ELDER STATUS:\nNone yet (call just started).")

        # Unknown status fields that still need to be asked
        priority_keys = ["medicationTaken", "breakfast", "lunch", "sleep", "painPresent", "dizziness", "mood"]
        missing_keys = [k for k in priority_keys if session.elderStatus.get(k) is None]
        if missing_keys:
            context_parts.append(f"UNKNOWN STATUS FIELDS NEEDING INQUIRY (pick the first unasked):\n{', '.join(missing_keys)}")

        if session.activeConcerns:
            concerns_formatted = "\n".join(
                f"- Level: {c.get('level')} | Reason: {c.get('reason')} | Action: {c.get('action')}"
                for c in session.activeConcerns[-3:]  # last 3 concerns
            )
            context_parts.append(f"ACTIVE CONCERNS:\n{concerns_formatted}")

        if (session and getattr(session, 'language', 'en').startswith("hi")) or any('\u0900' <= ch <= '\u097f' for ch in current_user_message):
            context_parts.append("LANGUAGE DIRECTIVE: Conversing in HINDI (हिन्दी). Provide your 'reply' in warm, polite, natural spoken Hindi using 'आप'. Output valid JSON with English keys as specified.")

        system_instruction_content = SYSTEM_PROMPT
        if context_parts:
            system_instruction_content += "\n\n=== CALL CONTEXT SO FAR ===\n" + "\n\n".join(context_parts)

        messages = [
            {"role": "system", "content": system_instruction_content}
        ]

        # Append recent turns (up to max turns)
        recent = session.recentTurns[-config.MAX_RECENT_TURNS:]
        for turn in recent:
            messages.append({"role": turn["role"], "content": turn["text"]})

        # Append current user utterance
        messages.append({"role": "user", "content": current_user_message})

        return messages

    @staticmethod
    def append_turn(session: SessionState, role: str, text: str, timestamp: str):
        """Append to transcript and recent turns."""
        turn = {"role": role, "text": text, "timestamp": timestamp}
        session.transcript.append(turn)
        session.recentTurns.append({"role": role, "text": text})

        # Keep recentTurns bounded
        if len(session.recentTurns) > config.MAX_RECENT_TURNS * 2:
            # Shift older turns into rolling summary representation
            older_turn = session.recentTurns.pop(0)
            # Lightweight addition to summary if not already summarized
            if len(session.rollingSummary) < 1500:
                session.rollingSummary += f" [{older_turn['role']}: {older_turn['text']}]"
