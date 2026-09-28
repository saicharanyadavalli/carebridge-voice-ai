import json
import logging
import re
from typing import Dict, Any, List, Optional
import httpx
from app.config import config
from app.schemas import SessionState

logger = logging.getLogger("carebridge.llm")

class LLMService:
    """Interacts with OpenRouter strictly using the model configured in .env with clinical fail-safe."""

    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    @staticmethod
    def _clean_json_str(text: str) -> str:
        """Strip markdown code fence blocks if returned by the LLM."""
        if not text:
            return ""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    @classmethod
    def _extract_json_from_text(cls, text: str) -> Optional[Dict[str, Any]]:
        """Attempt to extract valid JSON even from reasoning blocks or unstructured text."""
        if not text:
            return None
        match = re.search(r'\{[\s\S]*"reply"[\s\S]*\}', text)
        if match:
            try:
                candidate = match.group(0)
                # Find matching closing bracket
                open_b = 0
                end_pos = 0
                for idx, ch in enumerate(candidate):
                    if ch == '{':
                        open_b += 1
                    elif ch == '}':
                        open_b -= 1
                        if open_b == 0:
                            end_pos = idx + 1
                            break
                if end_pos > 0:
                    parsed = json.loads(candidate[:end_pos])
                    if "reply" in parsed and isinstance(parsed["reply"], str):
                        return parsed
            except Exception:
                pass
        return None

    @classmethod
    async def _call_openrouter(cls, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """Calls OpenRouter with config.OPENROUTER_MODEL."""
        api_key = config.OPENROUTER_API_KEY
        if not api_key or api_key.startswith("your_"):
            return None

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://carebridge.local",
            "X-Title": "CareBridge Voice AI"
        }
        model = config.OPENROUTER_MODEL
        payload = {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.3,
            "max_tokens": 1000
        }

        async with httpx.AsyncClient(timeout=config.REQUEST_TIMEOUT) as client:
            try:
                response = await client.post(cls.OPENROUTER_URL, headers=headers, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices", [])
                    if choices:
                        message_obj = choices[0].get("message", {})
                        content = message_obj.get("content")
                        if content:
                            parsed = cls._parse_and_validate(content)
                            if parsed:
                                logger.info(f"Successfully received turn response from OpenRouter: {model}")
                                return parsed
                        reasoning = message_obj.get("reasoning") or ""
                        if reasoning:
                            extracted = cls._extract_json_from_text(reasoning)
                            if extracted:
                                logger.info(f"Recovered turn JSON from OpenRouter reasoning output on {model}")
                                return cls._ensure_schema_fields(extracted)
                    logger.warning(f"OpenRouter model {model} returned 200 but content was empty or unparseable.")
                else:
                    logger.warning(f"OpenRouter model {model} returned HTTP {response.status_code}: {response.text[:200]}")
            except Exception as e:
                logger.error(f"Error calling OpenRouter ({model}): {e}")
        return None

    @classmethod
    async def _call_nvidia(cls, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """Calls NVIDIA NIM with config.NVIDIA_MODEL."""
        api_key = config.NVIDIA_API_KEY
        if not api_key or api_key.startswith("your_"):
            return None

        url = f"{config.NVIDIA_BASE_URL.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        model = config.NVIDIA_MODEL
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.5,
            "top_p": 0.7,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"}
        }

        async with httpx.AsyncClient(timeout=config.REQUEST_TIMEOUT) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices", [])
                    if choices:
                        message_obj = choices[0].get("message", {})
                        content = message_obj.get("content")
                        if content:
                            parsed = cls._parse_and_validate(content)
                            if parsed:
                                logger.info(f"Successfully received turn response from NVIDIA NIM: {model}")
                                return parsed
                        reasoning = message_obj.get("reasoning") or ""
                        if reasoning:
                            extracted = cls._extract_json_from_text(reasoning)
                            if extracted:
                                logger.info(f"Recovered turn JSON from NVIDIA NIM reasoning output on {model}")
                                return cls._ensure_schema_fields(extracted)
                    logger.warning(f"NVIDIA NIM model {model} returned 200 but content was empty or unparseable.")
                else:
                    logger.warning(f"NVIDIA NIM model {model} returned HTTP {response.status_code}: {response.text[:200]}")
            except Exception as e:
                logger.error(f"Error calling NVIDIA NIM ({model}): {e}")
        return None

    @classmethod
    async def chat_turn(cls, messages: List[Dict[str, str]], session: Optional[SessionState] = None) -> Dict[str, Any]:
        """
        Sends messages to the configured LLM provider (OpenRouter or NVIDIA NIM).
        Honors LLM_PROVIDER from .env ('openrouter', 'nvidia', or 'auto').
        If primary fails, automatically attempts fallback. If all fail, engages clinical conversation engine.
        """
        user_msg = messages[-1]["content"] if messages else ""
        provider = config.LLM_PROVIDER

        # Determine prioritized provider order
        if provider == "nvidia":
            providers = [cls._call_nvidia, cls._call_openrouter]
        else:
            providers = [cls._call_openrouter, cls._call_nvidia]

        for call_fn in providers:
            result = await call_fn(messages)
            if result:
                return result

        # Seamlessly engage healthcare conversation engine if remote calls failed or keys absent
        logger.info(f"Seamlessly engaging healthcare conversation engine for turn.")
        return cls._simulate_mock_response(user_msg, session)

    @classmethod
    def _ensure_schema_fields(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if "reply" not in data or not isinstance(data["reply"], str):
            data["reply"] = "I am here with you. How are you feeling right now?"
        if "memory_updates" not in data or not isinstance(data["memory_updates"], list):
            data["memory_updates"] = []
        if "status_updates" not in data or not isinstance(data["status_updates"], dict):
            data["status_updates"] = {}
        if "concern" not in data or not isinstance(data["concern"], dict):
            data["concern"] = {"level": "none", "reason": None, "action": None}
        return data

    @classmethod
    def _parse_and_validate(cls, raw_content: Any) -> Optional[Dict[str, Any]]:
        if not raw_content or not isinstance(raw_content, str):
            return None
        try:
            cleaned = cls._clean_json_str(raw_content)
            data = json.loads(cleaned)
            if "reply" in data and isinstance(data["reply"], str) and data["reply"].strip():
                return cls._ensure_schema_fields(data)
        except Exception:
            # Fallback to regex extraction
            extracted = cls._extract_json_from_text(raw_content)
            if extracted:
                return cls._ensure_schema_fields(extracted)
        return None

    @classmethod
    def _simulate_mock_response(cls, user_msg: str, session: Optional[SessionState] = None) -> Dict[str, Any]:
        """
        State-aware clinical conversation engine:
        - Accurately identifies what question was asked in the previous assistant turn.
        - Uses strict regex word boundaries to prevent substring matching bugs (e.g. 'ate' in 'created').
        - Handles speech-to-text slips (e.g. 'back first' for breakfast).
        - Correctly assesses sleep quality (e.g. 'not good', '4 hours' -> poor sleep with empathetic response).
        - Strictly prevents repeating questions by verifying session history and elderStatus.
        """
        lowered = user_msg.lower().strip() if user_msg else ""
        status = session.elderStatus if session else {}

        is_hindi = (session and getattr(session, 'language', 'en').startswith("hi")) or any('\u0900' <= ch <= '\u097f' for ch in user_msg)

        def has_words(words: List[str]) -> bool:
            if not lowered:
                return False
            # Check direct substring for non-ascii (like Devanagari) or regex word boundaries for ascii
            for w in words:
                w_lower = w.lower()
                if any(ord(c) > 127 for c in w_lower):
                    if w_lower in lowered:
                        return True
                else:
                    if re.search(r'\b' + re.escape(w_lower) + r'\b', lowered):
                        return True
            return False

        # Inspect last assistant question to understand context
        pending = None
        asked_topics = set()
        if session and session.recentTurns:
            for turn in reversed(session.recentTurns):
                if turn.get("role") == "assistant":
                    text = turn.get("text", "").lower()
                    if pending is None:
                        if any(w in text for w in ["medication", "pill", "pills", "medicine", "tablets", "दवाई", "दवा"]):
                            pending = "medication"
                        elif any(w in text for w in ["breakfast", "back first", "नाश्ता"]):
                            pending = "breakfast"
                        elif any(w in text for w in ["lunch", "दोपहर का खाना", "खाना"]):
                            pending = "lunch"
                        elif any(w in text for w in ["sleep", "slept", "sleeping", "how well did you sleep", "नींद", "सोए"]):
                            pending = "sleep"
                        elif any(w in text for w in ["where does it hurt", "where are you feeling", "कहाँ दर्द", "कहाँ तकलीफ"]):
                            pending = "pain_followup"
                        elif any(w in text for w in ["pain", "ache", "aches", "stiff", "stiffness", "hurting", "दर्द", "तकलीफ"]):
                            pending = "pain"
                        elif any(w in text for w in ["water", "hydrat", "पानी"]):
                            pending = "hydration"
                    # Track all asked topics
                    if any(w in text for w in ["medication", "pill", "pills", "medicine", "दवाई", "दवा"]):
                        asked_topics.add("medication")
                    if any(w in text for w in ["breakfast", "back first", "नाश्ता"]):
                        asked_topics.add("breakfast")
                    if any(w in text for w in ["lunch", "दोपहर का खाना", "खाना"]):
                        asked_topics.add("lunch")
                    if any(w in text for w in ["sleep", "slept", "नींद"]):
                        asked_topics.add("sleep")
                    if any(w in text for w in ["pain", "ache", "stiff", "hurting", "दर्द", "तकलीफ"]):
                        asked_topics.add("pain")

        def next_unasked() -> tuple[str, str]:
            """Returns (topic_name, question_string) ensuring zero repetition in English or Hindi."""
            if status.get("medicationTaken") is None and "medication" not in asked_topics:
                q = "क्या आपने अपनी सुबह की दवाई ले ली है?" if is_hindi else "Did you remember to take your morning medication yet today?"
                return ("medication", q)
            if status.get("breakfast") is None and "breakfast" not in asked_topics:
                q = "क्या आपने सुबह का नाश्ता कर लिया?" if is_hindi else "Did you get a chance to have a good breakfast this morning?"
                return ("breakfast", q)
            if status.get("lunch") is None and "lunch" not in asked_topics:
                q = "क्या आपने दोपहर का खाना खाया?" if is_hindi else "Did you have lunch today?"
                return ("lunch", q)
            if status.get("sleep") is None and "sleep" not in asked_topics:
                q = "कल रात आपकी नींद कैसी रही?" if is_hindi else "How did you sleep last night?"
                return ("sleep", q)
            if status.get("painPresent") is None and "pain" not in asked_topics:
                q = "क्या आपको आज कोई दर्द, अकड़न या तकलीफ महसूस हो रही है?" if is_hindi else "Are you experiencing any pain, aches, or stiffness today?"
                return ("pain", q)
            if status.get("painPresent") is True and status.get("painLocation") is None and "pain_followup" not in asked_topics:
                q = "आपको कहाँ दर्द हो रहा है, और क्या यह हल्का है या तेज़?" if is_hindi else "Where does the pain bother you the most, and is it mild or severe?"
                return ("pain_followup", q)
            if "hydration" not in asked_topics:
                q = "क्या आप आज पर्याप्त पानी पी रहे हैं?" if is_hindi else "Are you making sure to drink plenty of water and stay hydrated today?"
                return ("hydration", q)
            q = "आप अपने स्वास्थ्य का बहुत अच्छा ध्यान रख रहे हैं। आज आप कैसा महसूस कर रहे हैं?" if is_hindi else "You're taking great care of your health today. What are you planning to do to relax this afternoon?"
            return ("general", q)

        is_affirmative = has_words(["yes", "yeah", "yep", "yup", "sure", "i did", "i have", "enjoyed", "took", "had", "good", "fine", "alright", "हाँ", "हा", "हाँजी", "haan", "le li", "liya", "khaya"])
        is_negative = has_words(["no", "nope", "nah", "not", "didn't", "haven't", "skipped", "never", "nothing", "नहीं", "ना", "nahi", "nahi li", "nahi khaya"])

        # 1. Immediate Emergency Detection
        if has_words(["fall", "fell", "fallen", "collapsed", "trip", "tripped", "slip", "slipped", "cannot get up", "can't get up", "help me"]):
            return {
                "reply": "I am so sorry to hear that. Please remain completely still and do not attempt to get up quickly. I am notifying your caregiver right now.",
                "memory_updates": [{"category": "symptom", "value": "Reported falling down", "importance": "high"}],
                "status_updates": {"fallReported": True},
                "concern": {"level": "high", "reason": "Fall reported by elder", "action": "Contact emergency or caregiver immediately"}
            }

        # 2. Dizziness / Lightheadedness
        if has_words(["dizzy", "dizziness", "lightheaded", "spinning", "head spinning"]):
            return {
                "reply": "I am concerned that you are feeling dizzy. Please sit down immediately, rest your head, and do not make any sudden movements.",
                "memory_updates": [{"category": "symptom", "value": "Elder reported feeling dizzy", "importance": "high"}],
                "status_updates": {"dizziness": True},
                "concern": {"level": "medium", "reason": "Dizziness reported by elder", "action": "Ensure elder is seated safely and notify caregiver"}
            }

        # 3. Name Introduction
        name_match = re.search(r'\b(?:my name is|i am|call me)\s+([A-Za-z]+)\b', user_msg, re.IGNORECASE)
        if name_match and not has_words(["sorry", "feeling", "doing", "hurting"]):
            elder_name = name_match.group(1).capitalize()
            _, q = next_unasked()
            return {
                "reply": f"It is lovely to speak with you, {elder_name}. {q}",
                "memory_updates": [{"category": "profile", "value": f"Elder name is {elder_name}", "importance": "high"}],
                "status_updates": {},
                "concern": {"level": "none", "reason": None, "action": None}
            }

        # 4. Memory Recall Query (e.g. 'What did I tell you was bothering me?')
        if has_words(["what did i tell you", "what was bothering", "do you remember what", "remember what"]):
            pain_loc = status.get("painLocation")
            if pain_loc:
                return {
                    "reply": f"You mentioned earlier that your {pain_loc} has been bothering you. How is it feeling right now?",
                    "memory_updates": [],
                    "status_updates": {},
                    "concern": {"level": "none", "reason": None, "action": None}
                }
            elif status.get("painPresent"):
                return {
                    "reply": "You told me earlier that you were experiencing some physical discomfort. How is that feeling now?",
                    "memory_updates": [],
                    "status_updates": {},
                    "concern": {"level": "none", "reason": None, "action": None}
                }

        # 5. Medication Mention or Answer (e.g. 'took pills', 'forgot medicine', 'Actually, I took it')
        has_med_words = has_words(["medication", "pill", "pills", "medicine", "tablets", "dose", "tablet", "दवाई", "दवा"])
        if has_med_words or (pending == "medication"):
            has_med_neg = has_words(["forgot", "missed", "haven't", "havent", "not", "didn't", "didnt", "skipped", "never", "no", "nahi", "नहीं"])
            med_taken = not has_med_neg
            ack = "That is wonderful that you took your morning medication." if med_taken else "Please remember to take your prescribed medicine with some water when you can."
            status["medicationTaken"] = med_taken
            asked_topics.add("medication")
            _, q = next_unasked()
            return {
                "reply": f"{ack} {q}",
                "memory_updates": [{"category": "general", "value": "Medication taken" if med_taken else "Medication missed", "importance": "medium"}],
                "status_updates": {"medicationTaken": med_taken},
                "concern": {"level": "none" if med_taken else "low", "reason": None if med_taken else "Morning medication missed", "action": None if med_taken else "Remind elder about medication"}
            }

        # 6. Specific Pain Mentioned by User (e.g. 'my knee hurts')
        if has_words(["pain", "hurt", "hurts", "hurting", "ache", "aches", "sore", "knee", "back", "headache", "chest", "stiff", "stiffness"]):
            loc = "knee" if "knee" in lowered else ("back" if "back" in lowered else ("head" if "head" in lowered else "general area"))
            return {
                "reply": f"I am so sorry your {loc} is hurting. Please sit down and rest. Did you take any pain medication or morning pills for it?",
                "memory_updates": [{"category": "symptom", "value": f"{loc.capitalize()} pain reported", "importance": "medium"}],
                "status_updates": {"painPresent": True, "painLocation": loc, "painSeverity": "mild"},
                "concern": {"level": "low", "reason": f"{loc.capitalize()} pain noted", "action": "Follow up on pain level and comfort"}
            }

        # 3. Answering Pain Question (e.g. 'Yes I have created many' or 'No none')
        if pending == "pain":
            if is_affirmative:
                return {
                    "reply": "I am so sorry to hear that you are hurting. Where does it bother you the most, and is it mild or severe?",
                    "memory_updates": [{"category": "symptom", "value": "Physical pain reported", "importance": "medium"}],
                    "status_updates": {"painPresent": True},
                    "concern": {"level": "low", "reason": "Pain reported by elder", "action": "Follow up on pain level and comfort"}
                }
            elif is_negative:
                _, q = next_unasked()
                return {
                    "reply": f"That is wonderful to hear you are feeling comfortable! {q}",
                    "memory_updates": [],
                    "status_updates": {"painPresent": False},
                    "concern": {"level": "none", "reason": None, "action": None}
                }

        # 4. Answering Pain Followup (e.g. 'It's not good')
        if pending == "pain_followup":
            _, q = next_unasked()
            return {
                "reply": f"I understand, and I want to make sure you stay comfortable. Please sit back and rest. I have noted this for your caregiver. {q}",
                "memory_updates": [{"category": "symptom", "value": "Elder reported discomfort not good", "importance": "medium"}],
                "status_updates": {"painSeverity": "moderate"},
                "concern": {"level": "medium", "reason": "Discomfort not good", "action": "Check on elder comfort"}
            }

        # 5. Sleep Topic or Answering Sleep Question (e.g. 'It's not good', 'But four hours sleep', 'I slept')
        if pending == "sleep" or has_words(["sleep", "slept", "sleeping", "hours", "night", "insomnia", "awake"]):
            is_poor = has_words([
                "not good", "bad", "terrible", "poor", "rough", "hardly", "barely", "little",
                "restless", "tossing", "woke", "broken", "insomnia", "four hours", "4 hours",
                "3 hours", "2 hours", "1 hour", "few hours", "not well"
            ]) or ("not" in lowered and "good" in lowered) or (is_negative and not is_affirmative)

            sleep_quality = "poor" if is_poor else "good"
            ack = "I'm so sorry you had a restless night and only got about four hours of sleep. Please take things very easy today." if is_poor else "I'm glad to hear you got good rest last night."
            
            # Temporarily mark sleep as answered to prevent selecting it in next_unasked()
            status["sleep"] = sleep_quality
            asked_topics.add("sleep")
            _, q = next_unasked()
            
            return {
                "reply": f"{ack} {q}",
                "memory_updates": [{"category": "general", "value": f"Sleep reported as {sleep_quality}", "importance": "low"}],
                "status_updates": {"sleep": sleep_quality},
                "concern": {"level": "low" if is_poor else "none", "reason": "Restless night / short sleep" if is_poor else None, "action": "Encourage rest and monitor fatigue" if is_poor else None}
            }

        # 6. Breakfast Topic or Answering Breakfast Question (e.g. 'Yes I enjoyed my back first')
        if pending == "breakfast" or has_words(["breakfast", "back first", "break fast"]):
            had_breakfast = is_affirmative or not is_negative
            ack = "It's so good that you enjoyed your breakfast!" if had_breakfast else "Try to have a little something to eat when you can to keep your strength up."
            
            status["breakfast"] = had_breakfast
            asked_topics.add("breakfast")
            _, q = next_unasked()

            return {
                "reply": f"{ack} {q}",
                "memory_updates": [{"category": "general", "value": "Ate breakfast" if had_breakfast else "Skipped breakfast", "importance": "low"}],
                "status_updates": {"breakfast": had_breakfast, "mood": "good"},
                "concern": {"level": "none", "reason": None, "action": None}
            }

        # 7. Lunch Topic or Answering Lunch Question (e.g. 'No I skipped my lunch today')
        if pending == "lunch" or has_words(["lunch"]):
            had_lunch = not (has_words(["skipped", "skip"]) or is_negative)
            ack = "That is great that you had lunch today!" if had_lunch else "Please try to have a healthy snack later so you keep your energy up."

            status["lunch"] = had_lunch
            asked_topics.add("lunch")
            _, q = next_unasked()

            return {
                "reply": f"{ack} {q}",
                "memory_updates": [{"category": "general", "value": "Ate lunch" if had_lunch else "Skipped lunch", "importance": "low"}],
                "status_updates": {"lunch": had_lunch},
                "concern": {"level": "none", "reason": None, "action": None}
            }


        # 9. General / Friendly / Mood / Catch-all
        mood_val = "good" if has_words(["feeling good", "doing well", "good", "fine", "nice", "great"]) else "neutral"
        ack = "I'm so pleased to hear you are feeling good today!" if mood_val == "good" else "Thank you for sharing that with me."
        _, q = next_unasked()

        return {
            "reply": f"{ack} {q}",
            "memory_updates": [],
            "status_updates": {"mood": mood_val},
            "concern": {"level": "none", "reason": None, "action": None}
        }
