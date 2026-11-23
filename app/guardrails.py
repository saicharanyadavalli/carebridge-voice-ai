import re
from typing import Dict, Any, Optional
from app.schemas import SessionState

class SafetyGuardrail:
    """
    Sub-millisecond Deterministic Clinical Safety Gate:
    Runs before any LLM execution to intercept critical emergencies, falls,
    and acute distress without risking LLM hallucinations or schema failures.
    """

    FALL_KEYWORDS = [
        "fall", "fell", "fallen", "collapsed", "trip", "tripped",
        "slip", "slipped", "cannot get up", "can't get up", "help me",
        "गिर गया", "गिर गई", "चोट लग गई", "उठ नहीं पा रहा"
    ]

    CHEST_PAIN_KEYWORDS = [
        "chest pain", "pressure in chest", "heart attack", "can't breathe",
        "cannot breathe", "shortness of breath", "सीने में दर्द", "सांस नहीं आ रही"
    ]

    DIZZY_KEYWORDS = [
        "dizzy", "dizziness", "lightheaded", "spinning", "head spinning",
        "चक्कर आ रहे", "सिर घूम रहा"
    ]

    @classmethod
    def check_emergency(cls, user_text: str, session: Optional[SessionState] = None) -> Optional[Dict[str, Any]]:
        """
        Scans user utterance for immediate life-safety emergencies.
        Returns emergency response dict if triggered, otherwise None.
        """
        if not user_text:
            return None

        text_lower = user_text.lower().strip()
        is_hindi = (session and getattr(session, 'language', 'en').startswith("hi")) or any('\u0900' <= ch <= '\u097f' for ch in user_text)

        def matches_any(keywords):
            for kw in keywords:
                if any(ord(c) > 127 for c in kw):
                    if kw in text_lower:
                        return True
                else:
                    if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                        return True
            return False

        # 1. Fall / Inability to get up (Highest Priority)
        if matches_any(cls.FALL_KEYWORDS):
            reply = (
                "कृपया बिल्कुल शांत रहें और तुरंत उठने की कोशिश न करें। मैं तुरंत आपके परिवार और देखभालकर्ता को सूचित कर रहा हूँ।"
                if is_hindi else
                "I am so sorry to hear that. Please remain completely still and do not attempt to get up quickly. I am notifying your caregiver right now."
            )
            return {
                "triggered": True,
                "reply": reply,
                "status_updates": {"fallReported": True},
                "memory_updates": [{"category": "symptom", "value": "Reported falling down", "importance": "high"}],
                "concern": {
                    "level": "high",
                    "reason": "Fall reported by elder",
                    "action": "Contact emergency services and caregiver immediately"
                }
            }

        # 2. Acute Chest Pain / Respiratory Distress
        if matches_any(cls.CHEST_PAIN_KEYWORDS):
            reply = (
                "सीने में दर्द या सांस लेने में तकलीफ बहुत गंभीर हो सकती है। कृपया तुरंत बैठ जाएं, आराम करें और किसी को बुलाएं। मैं सहायता के लिए सूचना भेज रहा हूँ।"
                if is_hindi else
                "Chest pain or difficulty breathing is very serious. Please sit down immediately, rest comfortably, and do not exert yourself. I am alerting your caregiver right now."
            )
            return {
                "triggered": True,
                "reply": reply,
                "status_updates": {"painPresent": True, "painLocation": "chest", "painSeverity": "severe"},
                "memory_updates": [{"category": "symptom", "value": "Acute chest pain or breathing difficulty", "importance": "high"}],
                "concern": {
                    "level": "high",
                    "reason": "Acute chest pain or breathing difficulty reported",
                    "action": "Immediate medical attention required"
                }
            }

        # 3. Sudden Dizziness / Near-syncope
        if matches_any(cls.DIZZY_KEYWORDS):
            reply = (
                "मुझे चिंता है कि आपको चक्कर आ रहे हैं। कृपया तुरंत बैठ जाएं, सिर को सहारा दें और अचानक कोई हरकत न करें।"
                if is_hindi else
                "I am concerned that you are feeling dizzy. Please sit down immediately, rest your head, and do not make any sudden movements."
            )
            return {
                "triggered": True,
                "reply": reply,
                "status_updates": {"dizziness": True},
                "memory_updates": [{"category": "symptom", "value": "Elder reported feeling dizzy", "importance": "high"}],
                "concern": {
                    "level": "medium",
                    "reason": "Dizziness reported by elder",
                    "action": "Ensure elder is seated safely and notify caregiver"
                }
            }

        return None
