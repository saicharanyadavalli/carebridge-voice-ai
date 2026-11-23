from typing import List, Dict, Any
from app.schemas import SessionState
from app.config import config
from app.dialogue_engine.fsm import DialogueFSM

SYSTEM_PROMPT = """You are CareBridge, an empathetic, caring English voice companion dedicated to checking on an elderly person's daily health and routine.

CORE CONVERSATIONAL PROTOCOL:
You must proactively and warmly check on their essential daily health routine by asking ONE targeted question at a time.
Inquiry priority sequence:
1. MEDICATIONS: Ask if they remembered to take their prescribed morning or daily pills/medication.
2. MEALS: Ask if they had breakfast, lunch, or a proper meal today.
3. SLEEP & REST: Ask how they slept through the night.
4. PHYSICAL COMFORT & PAIN: Ask if they are experiencing any pain, aches, knee/back stiffness, or dizziness.
5. HYDRATION: Ask if they are drinking enough water throughout the day.
6. GENERAL MOOD & WELLBEING: Ask about their mood, feelings, or plans to relax.
7. WRAP-UP & GRACEFUL CONCLUSION:
   - Once all the health check items above are completed, ask politely: "Is there anything else you would like to discuss or need help with today?"
   - IF the elder says "no", "nothing", "all good", "that's all", "bye", "thank you", or indicates they have nothing more to ask:
     * Conclude the call warmly and politely (e.g. "Thank you so much for chatting with me today. Take good care of yourself, and I will prepare your summary now. Goodbye!").
     * Set "call_ended": true in the output JSON.
   - IF the elder asks an additional question:
     * If it is relevant to their daily health, routine, or safety, answer concisely and warmly, then ask again: "Is there anything else I can help you with today?" Keep "call_ended": false.
     * CRITICAL RELEVANCE GUARDRAIL: Do NOT answer irrelevant, off-topic, or non-critical questions (e.g. internet trivia, politics, complex general tech). Politely steer back: "I am here focused on your daily health and comfort. Is there anything about how you're feeling or your routine you'd like to share?"

CRITICAL RULES:
- Ask only ONE short, warm question per turn (1-2 sentences max).
- CONVERSATIONAL CONTEXT: The elder's utterance is answering the question you JUST asked in the previous turn.
- SLEEP EMPATHY: If the elder reports poor or short sleep (e.g. 'not good', '4 hours', 'restless'), record sleep as 'poor' and respond with empathy. NEVER say 'I'm glad to hear you got good rest' if they reported poor or short sleep!
- NEVER REPEAT: Never re-ask questions that were already asked or recorded.
- Never diagnose conditions or claim to be a doctor.
- If pain, a fall, or dizziness is reported, address it with care and record it in concern.

You MUST reply with ONLY a valid JSON object adhering to this schema:
{
  "reply": "natural spoken response for text-to-speech (warm, concise, 1-2 sentences)",
  "call_ended": false,
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
        2. Dialogue State Machine Target Objective (zero-repetition clinical directive)
        3. Recent conversation turns (last 4-8 turns)
        4. Current user message
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

        # Dialogue State Machine target objective
        next_topic, q_en, q_hi = DialogueFSM.determine_next_objective(session)
        pending_topic = DialogueFSM.get_pending_question(session)
        if pending_topic:
            context_parts.append(f"PENDING INQUIRY ANSWERED IN THIS TURN:\nThe elder's message is answering your previous question about: {pending_topic}")
        context_parts.append(f"NEXT TARGET CONVERSATIONAL TOPIC (DO NOT REPEAT PREVIOUS QUESTIONS):\nInquire next about: {next_topic} (Recommended: '{q_en}')")

        # Time of day context & prioritized unknown status fields
        hour = DialogueFSM.get_current_hour(session)
        period = DialogueFSM.get_time_period(hour)
        context_parts.append(
            f"TIME OF DAY CONTEXT: Current time period is {period.upper()} (hour: {hour:02d}:00). "
            f"Always inquire about meals and health appropriate for {period.upper()} "
            f"(e.g. breakfast in the morning, lunch in the afternoon, dinner in the evening/night)."
        )

        protocol_seq = DialogueFSM.get_protocol_sequence(session)
        missing_keys = [k for k in protocol_seq if k != "wrap_up" and session.elderStatus.get(k) is None]
        if missing_keys:
            context_parts.append(f"UNKNOWN STATUS FIELDS NEEDING INQUIRY (prioritized for {period}):\n{', '.join(missing_keys)}")

        if session.activeConcerns:
            concerns_formatted = "\n".join(
                f"- Level: {c.get('level')} | Reason: {c.get('reason')} | Action: {c.get('action')}"
                for c in session.activeConcerns[-3:]  # last 3 concerns
            )
            context_parts.append(f"ACTIVE CONCERNS:\n{concerns_formatted}")

        if (session and getattr(session, 'language', 'en').startswith("hi")) or any('\u0900' <= ch <= '\u097f' for ch in current_user_message):
            context_parts.append(
                "LANGUAGE DIRECTIVE — HIGH QUALITY HINDI (हिन्दी):\n"
                "- You MUST converse strictly in natural, warm, conversational HINDI (Devanagari script).\n"
                "- Address the elder respectfully using 'आप', 'आपका', and affectionate, polite terms like 'जी' where natural.\n"
                "- Your spoken 'reply' must be 1-2 short, crystal-clear sentences (optimized for Text-to-Speech audio).\n"
                "- Speak like an empathetic, caring family member or elder-care companion, avoiding robotic translations.\n"
                "- When wrap-up question is asked and elder says 'नहीं', 'कुछ नहीं', 'बस इतना ही', 'धन्यवाद', say a warm goodbye ('आज बात करने के लिए धन्यवाद जी, अपना ध्यान रखिएगा! नमस्ते!') and set 'call_ended': true.\n"
                "- If the elder asks an irrelevant or off-topic question, gently steer back to their health/routine.\n"
                "- CRITICAL: Output must remain valid JSON with English keys ('reply', 'call_ended', 'memory_updates', 'status_updates', 'concern')."
            )

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
