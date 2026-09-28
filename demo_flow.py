import asyncio
import json
from app.session_store import session_store
from app.context_manager import ContextManager
from app.llm_service import LLMService
from app.status_manager import StatusManager
from app.memory_manager import MemoryManager
from app.concern_manager import ConcernManager
from app.summary_service import SummaryService

async def run_demonstration():
    print("="*60)
    print("CAREBRIDGE VERIFICATION & DEMONSTRATION RUN")
    print("="*60)

    # 1. Start Session
    session = session_store.create_session()
    call_id = session.callId
    initial_greeting = "Hello! I am CareBridge, your voice companion. It's so nice to speak with you today. How are you feeling, and did you remember to take your morning medication yet?"
    ContextManager.append_turn(session, "assistant", initial_greeting, "2026-09-27T12:00:00Z")

    print("\n--- TURN 0: Call Started ---")
    print(f"AI GREETING: \"{initial_greeting}\"")

    # Conversation Demonstration Script
    script = [
        "Yes, I took my blood pressure pill with a glass of water.",
        "I had two slices of toast and some tea for breakfast.",
        "I didn't sleep very well last night, was tossing and turning.",
        "My left knee has been hurting a bit when I walk down the stairs."
    ]

    for idx, user_input in enumerate(script, start=1):
        print(f"\n--- TURN {idx} ---")
        print(f"USER INPUT:  \"{user_input}\"")

        # Record user turn
        ContextManager.append_turn(session, "user", user_input, f"2026-09-27T12:0{idx}:00Z")

        # Build compact context
        messages = ContextManager.build_compact_context(session, user_input)

        # Get response
        result = await LLMService.chat_turn(messages)
        ai_reply = result.get("reply", "")
        print(f"AI OUTPUT:   \"{ai_reply}\"")

        # Apply updates
        if result.get("status_updates"):
            StatusManager.update_status(session, result["status_updates"])
        if result.get("memory_updates"):
            MemoryManager.add_memories(session, result["memory_updates"])
        if result.get("concern"):
            ConcernManager.process_concern(session, result["concern"])

        ContextManager.append_turn(session, "assistant", ai_reply, f"2026-09-27T12:0{idx}:30Z")

        print("UPDATED STATUS:", {k: v for k, v in session.elderStatus.items() if v is not None})
        if session.activeConcerns:
            print("ACTIVE CONCERNS:", session.activeConcerns[-1])

    # End Session and Generate Summary
    session_store.end_session(call_id)
    summary = await SummaryService.generate_summary(session)

    print("\n" + "="*60)
    print("CAREGIVER END-OF-CALL SUMMARY REPORT")
    print("="*60)
    print("OVERALL STATUS:", summary.get("overall_status"))
    print("SUMMARY TEXT:  ", summary.get("summary"))
    print("KEY POINTS:    ", json.dumps(summary.get("key_points"), indent=2))
    print("CONCERNS:      ", json.dumps(summary.get("concerns"), indent=2))
    print("CAREGIVER ACTIONS:", json.dumps(summary.get("caregiver_actions"), indent=2))
    print("="*60)

if __name__ == "__main__":
    asyncio.run(run_demonstration())
