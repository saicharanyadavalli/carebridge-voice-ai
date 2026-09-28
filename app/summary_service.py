import json
import logging
from typing import Dict, Any, List
import httpx
from app.config import config
from app.schemas import SessionState, FinalSummaryResponse

logger = logging.getLogger("carebridge.summary")

SUMMARY_PROMPT = """You are CareBridge clinical caregiver summarizer.
Produce a structured, objective, empathetic end-of-call summary for a caregiver based on the elder's conversation session.
Do not invent facts. Accurately reflect health status, pain, sleep, medications, and concerns noted.

You MUST reply with ONLY a JSON object with this exact schema:
{
  "summary": "1-3 sentences describing the overall call and elder condition.",
  "overall_status": "stable" | "attention_needed" | "urgent",
  "key_points": ["point 1", "point 2"],
  "elder_status": {},
  "concerns": [{"level": "low|medium|high", "issue": "description"}],
  "caregiver_actions": ["concrete action for caregiver"]
}"""

class SummaryService:
    """Generates comprehensive caregiver summary at the end of a session."""

    @classmethod
    async def generate_summary(cls, session: SessionState) -> Dict[str, Any]:
        """
        Builds a compact summary payload from the session state and calls OpenRouter
        or generates a deterministic summary from recorded status and concerns.
        """
        # Determine baseline overall status
        status_rank = "stable"
        if any(c.get("level") == "high" for c in session.activeConcerns) or session.elderStatus.get("fallReported") is True:
            status_rank = "urgent"
        elif any(c.get("level") in ("medium", "low") for c in session.activeConcerns) or session.elderStatus.get("painPresent") is True:
            status_rank = "attention_needed"

        # Check if API key is present for LLM generation
        if not config.OPENROUTER_API_KEY or config.OPENROUTER_API_KEY.startswith("your_"):
            return cls._build_heuristic_summary(session, status_rank)

        # Prepare context payload for summary
        call_details = {
            "duration": f"Started {session.startedAt}, ended {session.endedAt}",
            "elderStatus": session.elderStatus,
            "importantMemory": session.importantMemory,
            "activeConcerns": session.activeConcerns,
            "rollingSummary": session.rollingSummary,
            "recentTurns": session.recentTurns
        }

        messages = [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": f"Summarize this elder session:\n{json.dumps(call_details, default=str)}"}
        ]

        # Attempt LLM summary generation with prioritized providers
        provider_calls = []
        if config.LLM_PROVIDER == "nvidia":
            if config.NVIDIA_API_KEY and not config.NVIDIA_API_KEY.startswith("your_"):
                provider_calls.append(("NVIDIA NIM", f"{config.NVIDIA_BASE_URL.rstrip('/')}/chat/completions", {"Authorization": f"Bearer {config.NVIDIA_API_KEY}", "Content-Type": "application/json"}, config.NVIDIA_MODEL))
            if config.OPENROUTER_API_KEY and not config.OPENROUTER_API_KEY.startswith("your_"):
                provider_calls.append(("OpenRouter", "https://openrouter.ai/api/v1/chat/completions", {"Authorization": f"Bearer {config.OPENROUTER_API_KEY}", "Content-Type": "application/json", "HTTP-Referer": "https://carebridge.local"}, config.OPENROUTER_MODEL))
        else:
            if config.OPENROUTER_API_KEY and not config.OPENROUTER_API_KEY.startswith("your_"):
                provider_calls.append(("OpenRouter", "https://openrouter.ai/api/v1/chat/completions", {"Authorization": f"Bearer {config.OPENROUTER_API_KEY}", "Content-Type": "application/json", "HTTP-Referer": "https://carebridge.local"}, config.OPENROUTER_MODEL))
            if config.NVIDIA_API_KEY and not config.NVIDIA_API_KEY.startswith("your_"):
                provider_calls.append(("NVIDIA NIM", f"{config.NVIDIA_BASE_URL.rstrip('/')}/chat/completions", {"Authorization": f"Bearer {config.NVIDIA_API_KEY}", "Content-Type": "application/json"}, config.NVIDIA_MODEL))

        async with httpx.AsyncClient(timeout=config.REQUEST_TIMEOUT) as client:
            for p_name, p_url, p_headers, p_model in provider_calls:
                payload = {
                    "model": p_model,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                    "temperature": 0.2,
                    "max_tokens": 1200
                }
                try:
                    response = await client.post(p_url, headers=p_headers, json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        choices = data.get("choices", [])
                        if choices:
                            raw_content = choices[0].get("message", {}).get("content")
                            if raw_content and isinstance(raw_content, str):
                                content = raw_content.strip()
                                if content.startswith("```json"):
                                    content = content[7:]
                                if content.startswith("```"):
                                    content = content[3:]
                                if content.endswith("```"):
                                    content = content[:-3]
                                parsed = json.loads(content.strip())
                                if "summary" in parsed and "overall_status" in parsed:
                                    logger.info(f"Summary successfully generated via {p_name}")
                                    return parsed
                except Exception as e:
                    logger.error(f"Error generating summary via {p_name}: {e}")

        return cls._build_heuristic_summary(session, status_rank)

    @classmethod
    def _build_heuristic_summary(cls, session: SessionState, status_rank: str) -> Dict[str, Any]:
        """Provides a reliable, clinical summary directly from structured session state."""
        key_points = []
        concerns = []
        caregiver_actions = []

        es = session.elderStatus
        if es.get("mood"):
            key_points.append(f"Mood noted as: {es['mood']}.")
        if es.get("sleep"):
            key_points.append(f"Sleep reported as {es['sleep']}.")
        if es.get("medicationTaken") is True:
            key_points.append("Medication confirmed taken.")
        elif es.get("medicationTaken") is False:
            key_points.append("Medication was reported NOT taken.")
            caregiver_actions.append("Follow up to ensure necessary medication is taken.")

        if es.get("painPresent"):
            loc = es.get("painLocation") or "unspecified area"
            sev = es.get("painSeverity") or "mild"
            key_points.append(f"Pain reported: {sev} in {loc}.")
            caregiver_actions.append(f"Check in regarding {loc} discomfort.")

        if es.get("fallReported"):
            key_points.append("URGENT: Elder reported a fall during call.")
            concerns.append({"level": "high", "issue": "Fall reported"})
            caregiver_actions.insert(0, "Check on elder in-person or call emergency contact immediately.")

        for c in session.activeConcerns:
            concerns.append({"level": c.get("level", "low"), "issue": c.get("reason", "Concern noted")})
            if c.get("action"):
                caregiver_actions.append(c["action"])

        for m in session.importantMemory:
            key_points.append(f"{m.get('value')} ({m.get('category')})")

        if not key_points:
            key_points.append("Elder engaged in a brief check-in conversation.")

        summary_text = f"Call concluded with elder. Overall condition assessed as {status_rank.replace('_', ' ')}."
        if es.get("sleep") or es.get("painPresent"):
            summary_text += f" Elder reported sleep as {es.get('sleep', 'noted')} with pain: {'yes (' + str(es.get('painLocation')) + ')' if es.get('painPresent') else 'none'}."

        return {
            "summary": summary_text,
            "overall_status": status_rank,
            "key_points": list(dict.fromkeys(key_points))[:6],
            "elder_status": session.elderStatus,
            "concerns": concerns,
            "caregiver_actions": list(dict.fromkeys(caregiver_actions))[:5]
        }
