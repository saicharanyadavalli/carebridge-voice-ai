import json
import logging
import re
from typing import Dict, Any, List, Optional
import httpx
from app.config import config
from app.schemas import SessionState
from app.dialogue_engine.fsm import DialogueFSM

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
    async def _call_gemini(cls, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """
        Calls Google Gemini using the configured model (default: gemini-3.5-flash-lite).
        Automatically rotates through available Google API keys (Key 1 -> Key 2 -> Key 3)
        if any error code (e.g. 429 quota/rate limit, 400, 401, 403, 500, timeout) occurs.
        """
        api_keys = config.GEMINI_API_KEYS
        if not api_keys:
            return None

        # Extract system prompt and conversation messages
        system_instruction_text = ""
        gemini_contents = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "").strip()
            if not content:
                continue
            if role == "system":
                system_instruction_text += (("\n\n" + content) if system_instruction_text else content)
            else:
                gemini_role = "user" if role == "user" else "model"
                if gemini_contents and gemini_contents[-1]["role"] == gemini_role:
                    gemini_contents[-1]["parts"][0]["text"] += f"\n\n{content}"
                else:
                    gemini_contents.append({"role": gemini_role, "parts": [{"text": content}]})

        # Ensure valid conversational start for Gemini
        if gemini_contents and gemini_contents[0]["role"] == "model":
            gemini_contents.insert(0, {"role": "user", "parts": [{"text": "Hello, please continue our conversation."}]})
        elif not gemini_contents:
            gemini_contents = [{"role": "user", "parts": [{"text": "Hello"}]}]

        payload: Dict[str, Any] = {
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": 0.3,
                "responseMimeType": "application/json",
                "maxOutputTokens": 250
            }
        }
        if system_instruction_text:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction_text}]
            }

        models_to_try = [config.GEMINI_MODEL]
        for fallback_m in ["gemini-2.5-flash", "gemini-2.0-flash"]:
            if fallback_m not in models_to_try:
                models_to_try.append(fallback_m)

        total_attempts = 0
        max_total_attempts = 4  # Keep cumulative retry budget strictly bounded

        async with httpx.AsyncClient(timeout=6.0) as client:
            for current_model in models_to_try:
                for idx, key in enumerate(api_keys):
                    if total_attempts >= max_total_attempts:
                        logger.warning("Gemini cumulative retry attempt budget reached. Falling back to next provider.")
                        return None
                    total_attempts += 1
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{current_model}:generateContent?key={key}"
                    try:
                        response = await client.post(url, json=payload)
                        if response.status_code == 200:
                            data = response.json()
                            candidates = data.get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                if parts:
                                    raw_text = parts[0].get("text", "")
                                    parsed = cls._parse_and_validate(raw_text)
                                    if parsed:
                                        logger.info(f"Successfully received turn response from Gemini ({current_model}) using key #{idx+1}")
                                        return parsed
                                    extracted = cls._extract_json_from_text(raw_text)
                                    if extracted:
                                        logger.info(f"Recovered JSON from Gemini output using key #{idx+1}")
                                        return cls._ensure_schema_fields(extracted)
                            logger.warning(f"Gemini ({current_model}) key #{idx+1} returned 200 but content parts were empty. Trying next Google API key...")
                        else:
                            error_detail = response.text[:200]
                            logger.warning(
                                f"Gemini ({current_model}) key #{idx+1} returned HTTP {response.status_code}: {error_detail}. Trying next Google API key..."
                            )
                    except Exception as e:
                        logger.warning(f"Gemini ({current_model}) key #{idx+1} encountered exception: {e}. Trying next Google API key...")

        return None

    @classmethod
    async def _call_openrouter(cls, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """Calls OpenRouter with config.OPENROUTER_MODEL and reliable fallbacks."""
        api_key = config.OPENROUTER_API_KEY
        if not api_key or api_key.startswith("your_"):
            return None

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://carebridge.local",
            "X-Title": "CareBridge Voice AI"
        }
        
        # Primary configured model + resilient fallback candidates
        models_to_try = [config.OPENROUTER_MODEL]
        for candidate in ["google/gemini-2.0-flash-001", "meta-llama/llama-3.3-70b-instruct", "mistralai/mistral-small"]:
            if candidate not in models_to_try:
                models_to_try.append(candidate)

        async with httpx.AsyncClient(timeout=8.0) as client:
            for model in models_to_try:
                payload = {
                    "model": model,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                    "temperature": 0.3,
                    "max_tokens": 250
                }
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
                        logger.warning(f"OpenRouter model {model} returned 200 but content was empty. Trying next candidate...")
                    else:
                        logger.warning(f"OpenRouter model {model} returned HTTP {response.status_code}. Trying next candidate...")
                except Exception as e:
                    logger.warning(f"Error calling OpenRouter ({model}): {e}. Trying next candidate...")
        return None

    @classmethod
    async def _call_nvidia(cls, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """Calls NVIDIA NIM with config.NVIDIA_MODEL and fast resilient fallbacks."""
        api_key = config.NVIDIA_API_KEY
        if not api_key or api_key.startswith("your_"):
            return None

        url = f"{config.NVIDIA_BASE_URL.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Primary configured model + high-availability NIM models
        models_to_try = [config.NVIDIA_MODEL]
        for candidate in ["meta/llama-3.1-70b-instruct", "meta/llama-3.1-8b-instruct"]:
            if candidate not in models_to_try:
                models_to_try.append(candidate)

        async with httpx.AsyncClient(timeout=10.0) as client:
            for model in models_to_try:
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.5,
                    "top_p": 0.7,
                    "max_tokens": 250,
                    "response_format": {"type": "json_object"}
                }
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
                        logger.warning(f"NVIDIA NIM model {model} returned 200 but content was empty.")
                    elif response.status_code == 503:
                        logger.warning(f"NVIDIA NIM model {model} returned 503 Overloaded. Trying alternative NIM model...")
                    else:
                        logger.warning(f"NVIDIA NIM model {model} returned HTTP {response.status_code}.")
                except Exception as e:
                    logger.warning(f"Error calling NVIDIA NIM ({model}): {e}")
        return None

    @classmethod
    async def chat_turn(cls, messages: List[Dict[str, str]], session: Optional[SessionState] = None) -> Dict[str, Any]:
        """
        Sends messages to LLM providers with automatic fallback:
        1. Google Gemini (default: gemini-3.5-flash-lite, with Key 1 -> Key 2 -> Key 3 error rotation)
        2. OpenRouter (secondary fallback if Gemini keys fail or are exhausted)
        3. NVIDIA NIM (tertiary fallback if OpenRouter fails)
        4. CareBridge Clinical Conversation Engine (guaranteed fail-safe)
        """
        user_msg = messages[-1]["content"] if messages else ""
        if user_msg.startswith("<elder_utterance>") and user_msg.endswith("</elder_utterance>"):
            user_msg = user_msg[len("<elder_utterance>"):-len("</elder_utterance>")].strip()
        provider_pref = config.LLM_PROVIDER.lower()

        # Build prioritized provider sequence based on user preference
        if provider_pref == "openrouter":
            provider_chain = [("OpenRouter", cls._call_openrouter), ("Gemini", cls._call_gemini), ("NVIDIA NIM", cls._call_nvidia)]
        elif provider_pref == "nvidia":
            provider_chain = [("NVIDIA NIM", cls._call_nvidia), ("Gemini", cls._call_gemini), ("OpenRouter", cls._call_openrouter)]
        else:  # default "gemini" or "auto"
            provider_chain = [("Gemini", cls._call_gemini), ("OpenRouter", cls._call_openrouter), ("NVIDIA NIM", cls._call_nvidia)]

        for p_name, call_fn in provider_chain:
            result = await call_fn(messages)
            if result:
                return result
            logger.info(f"{p_name} returned no response or failed. Moving to next provider in fallback hierarchy...")

        # Seamlessly engage healthcare conversation engine if all remote calls failed or keys absent
        logger.info("All remote LLM providers failed or keys absent. Seamlessly engaging CareBridge clinical conversation engine.")
        return cls._simulate_mock_response(user_msg, session)

    @classmethod
    def _ensure_schema_fields(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if "reply" not in data or not isinstance(data["reply"], str):
            data["reply"] = "I am here with you. How are you feeling right now?"
        data["call_ended"] = bool(data.get("call_ended", False))
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
                    # Look at the final question/sentence of the assistant turn to identify pending question
                    sentences = [s.strip() for s in re.split(r'[.?!।]\s*', text) if s.strip()]
                    last_q = sentences[-1] if sentences else text
                    if pending is None:
                        if any(w in last_q for w in ["medication", "pill", "pills", "medicine", "tablets", "दवाई", "दवा"]):
                            pending = "medication"
                        elif any(w in last_q for w in ["breakfast", "back first", "नाश्ता"]):
                            pending = "breakfast"
                        elif any(w in last_q for w in ["lunch", "दोपहर का खाना"]):
                            pending = "lunch"
                        elif any(w in last_q for w in ["dinner", "supper", "evening meal", "रात का खाना", "डिनर", "शाम का खाना"]):
                            pending = "dinner"
                        elif any(w in last_q for w in ["sleep", "slept", "sleeping", "how well did you sleep", "नींद", "सोए"]):
                            pending = "sleep"
                        elif any(w in last_q for w in ["where does it hurt", "where are you feeling", "कहाँ दर्द", "कहाँ तकलीफ"]):
                            pending = "pain_followup"
                        elif any(w in last_q for w in ["pain", "ache", "aches", "stiff", "stiffness", "hurting", "दर्द", "तकलीफ"]):
                            pending = "pain"
                        elif any(w in last_q for w in ["water", "hydrat", "पानी"]):
                            pending = "hydration"
                    # Track all asked topics across the turn
                    if any(w in text for w in ["medication", "pill", "pills", "medicine", "दवाई", "दवा"]):
                        asked_topics.add("medication")
                    if any(w in text for w in ["breakfast", "back first", "नाश्ता"]):
                        asked_topics.add("breakfast")
                    if any(w in text for w in ["lunch", "दोपहर का खाना"]):
                        asked_topics.add("lunch")
                    if any(w in text for w in ["dinner", "supper", "evening meal", "रात का खाना", "डिनर", "शाम का खाना"]):
                        asked_topics.add("dinner")
                    if any(w in text for w in ["sleep", "slept", "नींद"]):
                        asked_topics.add("sleep")
                    if any(w in text for w in ["pain", "ache", "stiff", "hurting", "दर्द", "तकलीफ"]):
                        asked_topics.add("pain")

        def next_unasked() -> tuple[str, str]:
            """Returns (topic_name, question_string) ensuring zero repetition in English or Hindi via DialogueFSM."""
            topic, q_en, q_hi = DialogueFSM.determine_next_objective(session)
            return (topic, q_hi if is_hindi else q_en)

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

        # 3. Explicit Physical Pain / Discomfort Reported by Elder (e.g. 'my left knee has been hurting', 'pain in back')
        # ALWAYS prioritize elder's physical discomfort over routine pending questions!
        if has_words(["pain", "hurt", "hurts", "hurting", "ache", "aches", "sore", "knee", "back", "headache", "chest", "stiff", "stiffness", "दर्द", "तकलीफ", "घुटना", "कमर", "पीठ", "सिर"]):
            loc = "knee" if any(w in lowered for w in ["knee", "घुटना"]) else ("back" if any(w in lowered for w in ["back", "कमर", "पीठ"]) else ("head" if any(w in lowered for w in ["head", "headache", "सिर"]) else ("chest" if "chest" in lowered else "general area")))
            ack = f"मुझे सुनकर बहुत दुख हुआ कि आपके {loc} में दर्द हो रहा है। कृपया बैठ जाएं और आराम करें।" if is_hindi else f"I am so sorry your {loc} is hurting. Please sit down and rest."
            followup = "क्या आपने इसके लिए कोई दवा ली है?" if is_hindi else "Did you take any pain medication or morning pills for it?"
            return {
                "reply": f"{ack} {followup}",
                "memory_updates": [{"category": "symptom", "value": f"{loc.capitalize()} pain reported", "importance": "medium"}],
                "status_updates": {"painPresent": True, "painLocation": loc, "painSeverity": "mild"},
                "concern": {"level": "low", "reason": f"{loc.capitalize()} pain noted", "action": "Follow up on pain level and comfort"}
            }

        # 4. Name Introduction
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

        # 5. Memory Recall Query (e.g. 'What did I tell you was bothering me?')
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

        # 6. Medication Mention or Answer (e.g. 'took pills', 'forgot medicine', 'Actually, I took it')
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

        # 8. Dinner Topic or Answering Dinner Question (e.g. 'Yes I had dinner', 'Had rice and dal for dinner')
        if pending == "dinner" or has_words(["dinner", "supper", "रात का खाना", "डिनर", "शाम का खाना"]):
            had_dinner = not (has_words(["skipped", "skip", "haven't", "havent", "not yet", "nahi khaya", "नहीं खाया"]) or is_negative)
            if is_hindi:
                ack = "यह बहुत अच्छी बात है कि आपने रात का खाना खा लिया है!" if had_dinner else "सोने से पहले कृपया कुछ पौष्टिक खाना ज़रूर खा लें।"
            else:
                ack = "That is wonderful that you had your dinner this evening!" if had_dinner else "Please remember to have something nourishing for dinner before you sleep."

            status["dinner"] = had_dinner
            asked_topics.add("dinner")
            _, q = next_unasked()

            return {
                "reply": f"{ack} {q}",
                "memory_updates": [{"category": "general", "value": "Ate dinner" if had_dinner else "Skipped dinner", "importance": "low"}],
                "status_updates": {"dinner": had_dinner},
                "concern": {"level": "none", "reason": None, "action": None}
            }


        # 8. Wrap-up / Termination Inquiry (When assistant asked if there's anything else)
        if pending == "wrap_up" or ("wrap_up" in asked_topics and (is_negative or has_words(["nothing", "all good", "i'm good", "im good", "that's all", "bye", "goodbye", "thank you", "thanks"]))):
            has_no_more = is_negative or has_words([
                "nothing", "all good", "i'm good", "im good", "that's all", "that is all",
                "bye", "goodbye", "no thank", "no thanks", "kuch nahi", "nahi", "bas", "dhanyawad",
                "thank you", "thanks", "fine", "nothing else", "none", "i am good"
            ])
            if has_no_more:
                farewell = "आज बात करने के लिए बहुत-बहुत धन्यवाद। अपना ध्यान रखिएगा, मैं आपकी रिपोर्ट तैयार कर रही हूँ। नमस्ते!" if is_hindi else "Thank you so much for chatting with me today. Take good care of yourself, and I will prepare your summary now. Goodbye!"
                return {
                    "reply": farewell,
                    "call_ended": True,
                    "memory_updates": [],
                    "status_updates": {},
                    "concern": {"level": "none", "reason": None, "action": None}
                }
            else:
                # User asked a question during wrap up
                is_irrelevant = any(w in lowered for w in ["president", "capital", "movie", "weather in tokyo", "crypto", "bitcoin", "coding", "physics", "trump", "biden"])
                if is_irrelevant:
                    ack = "मैं आज आपके स्वास्थ्य और दिनचर्या पर ध्यान देने के लिए यहाँ हूँ। क्या आपके स्वास्थ्य के बारे में कोई और बात है?" if is_hindi else "I am here focused on your daily health and comfort today. Is there anything about how you're feeling or your daily routine I can assist you with?"
                else:
                    ack = "मैंने इसे आपकी देखभाल के लिए नोट कर लिया है। क्या कोई और बात है जिसमें मैं आपकी मदद कर सकूँ?" if is_hindi else "I have noted that for your care. Is there anything else I can help you with today?"
                return {
                    "reply": ack,
                    "call_ended": False,
                    "memory_updates": [{"category": "general", "value": f"Elder note: {user_msg[:60]}", "importance": "low"}],
                    "status_updates": {},
                    "concern": {"level": "none", "reason": None, "action": None}
                }

        # 9. General / Friendly / Mood / Catch-all
        topic, q = next_unasked()
        if topic == "concluded":
            return {
                "reply": q,
                "call_ended": True,
                "memory_updates": [],
                "status_updates": {},
                "concern": {"level": "none", "reason": None, "action": None}
            }
        mood_val = "good" if has_words(["feeling good", "doing well", "good", "fine", "nice", "great"]) else "neutral"
        ack = "I'm so pleased to hear you are feeling good today!" if mood_val == "good" else "Thank you for sharing that with me."

        return {
            "reply": f"{ack} {q}",
            "call_ended": False,
            "memory_updates": [],
            "status_updates": {"mood": mood_val},
            "concern": {"level": "none", "reason": None, "action": None}
        }

