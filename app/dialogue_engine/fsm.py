import re
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from app.schemas import SessionState
from app.dialogue_engine.protocol import ProtocolState

class DialogueFSM:
    """
    Deterministic Dialogue Finite State Machine (FSM):
    Enforces time-of-day contextual dialogue sequences (breakfast, lunch, dinner),
    guarantees zero question repetition, and isolates the active conversational objective.
    """

    QUESTION_TEMPLATES_EN = {
        "medicationTaken": "Did you remember to take your medication yet today?",
        "breakfast": "Did you get a chance to have a good breakfast this morning?",
        "lunch": "Did you get a chance to have your lunch today?",
        "dinner": "Did you get a chance to have your dinner or an evening meal yet?",
        "sleep": "How did you sleep last night?",
        "painPresent": "Are you experiencing any pain, aches, or stiffness today?",
        "pain_investigation": "Where does the pain bother you the most, and is it mild or severe?",
        "hydration": "Are you making sure to drink plenty of water and stay hydrated today?",
        "mood": "You're taking great care of your health today. How are you feeling overall?",
        "wrap_up": "Is there anything else you would like to discuss or need help with today?",
        "concluded": "Thank you so much for chatting with me today. Take good care of yourself, and I will prepare your summary now. Goodbye!"
    }

    QUESTION_TEMPLATES_HI = {
        "medicationTaken": "क्या आपने अपनी आज की दवाई ले ली है?",
        "breakfast": "क्या आपने सुबह का नाश्ता कर लिया?",
        "lunch": "क्या आपने दोपहर का खाना खा लिया है?",
        "dinner": "क्या आपने रात का खाना खा लिया है?",
        "sleep": "कल रात आपकी नींद कैसी रही?",
        "painPresent": "क्या आपको आज कोई दर्द, अकड़न या तकलीफ महसूस हो रही है?",
        "pain_investigation": "आपको कहाँ दर्द हो रहा है, और क्या यह हल्का है या तेज़?",
        "hydration": "क्या आप आज पर्याप्त पानी पी रहे हैं?",
        "mood": "आप अपने स्वास्थ्य का बहुत अच्छा ध्यान रख रहे हैं। आज आप कैसा महसूस कर रहे हैं?",
        "wrap_up": "क्या आप किसी और चीज़ के बारे में बात करना चाहते हैं, या आपको किसी और सहायता की आवश्यकता है?",
        "concluded": "आज बात करने के लिए बहुत-बहुत धन्यवाद। अपना ध्यान रखिएगा, मैं आपकी रिपोर्ट तैयार कर रही हूँ। नमस्ते!"
    }

    @classmethod
    def get_current_hour(cls, session: Optional[SessionState] = None) -> int:
        """Determines the current local hour (0-23) from session clientHour or local time."""
        if session and getattr(session, "clientHour", None) is not None:
            try:
                return int(session.clientHour)
            except (ValueError, TypeError):
                pass
        return datetime.now().hour

    @classmethod
    def get_time_period(cls, hour: int) -> str:
        """
        Categorizes hour (0-23):
        - morning: 05:00 - 11:59 (breakfast, morning meds)
        - afternoon: 12:00 - 16:59 (lunch, hydration)
        - evening: 17:00 - 21:59 (dinner, evening meds)
        - night: 22:00 - 04:59 (sleep, rest, comfort)
        """
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 22:
            return "evening"
        else:
            return "night"

    @classmethod
    def get_initial_greeting(cls, language: str, hour: Optional[int] = None) -> str:
        """Generates time-of-day contextual initial greeting in English or Hindi."""
        h = hour if hour is not None else datetime.now().hour
        period = cls.get_time_period(h)
        is_hi = language.lower().startswith("hi")

        if period == "morning":
            if is_hi:
                return "शुभ प्रभात! मैं CareBridge हूँ, आपकी आवाज़ साथी। आज आप कैसा महसूस कर रहे हैं, और क्या आपने अपनी सुबह की दवाई ले ली है?"
            return "Good morning! I am CareBridge, your voice companion. It's so nice to speak with you today. How are you feeling, and did you remember to take your morning medication yet?"
        elif period == "afternoon":
            if is_hi:
                return "नमस्ते! मैं CareBridge हूँ, आपकी आवाज़ साथी। आशा है आपका दिन अच्छा बीत रहा होगा। आप कैसा महसूस कर रहे हैं, और क्या आपने दोपहर का खाना खा लिया है?"
            return "Good afternoon! I am CareBridge, your voice companion. I hope your day is going well. How are you feeling, and did you get a chance to have lunch yet?"
        elif period == "evening":
            if is_hi:
                return "शुभ संध्या! मैं CareBridge हूँ, आपकी आवाज़ साथी। आज शाम आप कैसा महसूस कर रहे हैं, और क्या आपने रात का खाना खा लिया है?"
            return "Good evening! I am CareBridge, your voice companion. It's lovely to check in with you. How are you feeling tonight, and have you had your dinner yet?"
        else:  # night
            if is_hi:
                return "नमस्ते! मैं CareBridge हूँ, आपकी आवाज़ साथी। यह जानने के लिए संपर्क किया कि आज रात आप कैसा महसूस कर रहे हैं, और क्या आप आराम से हैं?"
            return "Hello! I am CareBridge, your voice companion. Checking in to see how you are feeling tonight, and if you are comfortable and ready for rest."

    @classmethod
    def get_protocol_sequence(cls, session: Optional[SessionState] = None) -> List[str]:
        """Dynamically adapts inquiry priority based on the elder's time of day."""
        hour = cls.get_current_hour(session)
        period = cls.get_time_period(hour)

        if period == "morning":
            return [
                "medicationTaken",
                "breakfast",
                "sleep",
                "painPresent",
                "hydration",
                "mood",
                "wrap_up"
            ]
        elif period == "afternoon":
            return [
                "lunch",
                "medicationTaken",
                "hydration",
                "painPresent",
                "mood",
                "sleep",
                "wrap_up"
            ]
        elif period == "evening":
            return [
                "dinner",
                "medicationTaken",
                "painPresent",
                "hydration",
                "mood",
                "sleep",
                "wrap_up"
            ]
        else:  # night
            return [
                "sleep",
                "dinner",
                "medicationTaken",
                "painPresent",
                "hydration",
                "mood",
                "wrap_up"
            ]

    @classmethod
    def get_asked_topics(cls, session: SessionState) -> set:
        """Inspects past assistant turns to strictly prevent re-asking any topic."""
        asked = set()
        if not session or not session.recentTurns:
            return asked

        for turn in session.recentTurns:
            if turn.get("role") == "assistant":
                text = turn.get("text", "").lower()
                if any(w in text for w in ["medication", "pill", "pills", "medicine", "दवाई", "दवा"]):
                    asked.add("medicationTaken")
                if any(w in text for w in ["breakfast", "back first", "नाश्ता"]):
                    asked.add("breakfast")
                if any(w in text for w in ["lunch", "दोपहर का खाना"]):
                    asked.add("lunch")
                if any(w in text for w in ["dinner", "supper", "evening meal", "रात का खाना", "डिनर", "शाम का खाना"]):
                    asked.add("dinner")
                if any(w in text for w in ["sleep", "slept", "sleeping", "नींद", "सोए"]):
                    asked.add("sleep")
                if any(w in text for w in ["where does it hurt", "where are you feeling", "कहाँ दर्द", "कहाँ तकलीफ"]):
                    asked.add("pain_investigation")
                elif any(w in text for w in ["pain", "ache", "aches", "stiff", "hurting", "दर्द", "तकलीफ"]):
                    asked.add("painPresent")
                if any(w in text for w in ["water", "hydrat", "पानी"]):
                    asked.add("hydration")
                if any(w in text for w in ["relax", "feel", "mood", "कैसा महसूस"]):
                    asked.add("mood")
                if any(w in text for w in ["anything else", "anything you would like", "discuss or need help", "किसी और", "कोई और बात", "सहायता की आवश्यकता"]):
                    asked.add("wrap_up")
        return asked

    @classmethod
    def get_pending_question(cls, session: SessionState) -> Optional[str]:
        """Identifies what specific question the assistant asked in the immediately preceding turn."""
        if not session or not session.recentTurns:
            return None

        # Look at the last assistant turn
        for turn in reversed(session.recentTurns):
            if turn.get("role") == "assistant":
                text = turn.get("text", "").lower()
                sentences = [s.strip() for s in re.split(r'[.?!।]\s*', text) if s.strip()]
                last_q = sentences[-1] if sentences else text

                if any(w in last_q for w in ["anything else", "anything you would like", "discuss or need help", "किसी और", "कोई और बात", "सहायता की आवश्यकता"]):
                    return "wrap_up"
                elif any(w in last_q for w in ["medication", "pill", "pills", "medicine", "tablets", "दवाई", "दवा"]):
                    return "medicationTaken"
                elif any(w in last_q for w in ["breakfast", "back first", "नाश्ता"]):
                    return "breakfast"
                elif any(w in last_q for w in ["lunch", "दोपहर का खाना"]):
                    return "lunch"
                elif any(w in last_q for w in ["dinner", "supper", "evening meal", "रात का खाना", "डिनर", "शाम का खाना"]):
                    return "dinner"
                elif any(w in last_q for w in ["sleep", "slept", "sleeping", "नींद", "सोए"]):
                    return "sleep"
                elif any(w in last_q for w in ["where does it hurt", "where are you feeling", "कहाँ दर्द", "कहाँ तकलीफ"]):
                    return "pain_investigation"
                elif any(w in last_q for w in ["pain", "ache", "aches", "stiff", "hurting", "दर्द", "तकलीफ"]):
                    return "painPresent"
                elif any(w in last_q for w in ["water", "hydrat", "पानी"]):
                    return "hydration"
                elif any(w in last_q for w in ["relax", "feel", "mood", "कैसा महसूस"]):
                    return "mood"
                break
        return None

    @classmethod
    def determine_next_objective(cls, session: SessionState) -> Tuple[str, str, str]:
        """
        Determines the single next conversational objective.
        Returns: (active_topic, question_en, question_hi)
        Ensures zero repetition by checking both elderStatus and asked topics against the time-adapted protocol.
        """
        status = session.elderStatus if session else {}
        asked = cls.get_asked_topics(session)

        # 1. Immediate priority: Symptom investigation if pain reported but location/severity unknown
        if status.get("painPresent") is True and not status.get("painLocation") and "pain_investigation" not in asked:
            return (
                "pain_investigation",
                cls.QUESTION_TEMPLATES_EN["pain_investigation"],
                cls.QUESTION_TEMPLATES_HI["pain_investigation"]
            )

        # 2. Sequential clinical protocol dynamically adapted to current time of day
        protocol = cls.get_protocol_sequence(session)
        for topic in protocol:
            if topic in ("wrap_up",):
                if topic not in asked:
                    return (
                        "wrap_up",
                        cls.QUESTION_TEMPLATES_EN["wrap_up"],
                        cls.QUESTION_TEMPLATES_HI["wrap_up"]
                    )
            elif status.get(topic) is None and topic not in asked:
                return (
                    topic,
                    cls.QUESTION_TEMPLATES_EN.get(topic, cls.QUESTION_TEMPLATES_EN["mood"]),
                    cls.QUESTION_TEMPLATES_HI.get(topic, cls.QUESTION_TEMPLATES_HI["mood"])
                )

        # 3. If wrap_up was already asked, concluding closing
        return (
            "concluded",
            cls.QUESTION_TEMPLATES_EN["concluded"],
            cls.QUESTION_TEMPLATES_HI["concluded"]
        )
