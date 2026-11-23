/**
 * CareBridge Voice Engine
 * Handles Browser SpeechRecognition (STT), SpeechSynthesis (TTS),
 * and the continuous conversational state machine.
 */

// API Base URL Resolver
const getApiBase = () => {
  const urlParams = new URLSearchParams(window.location.search);
  const paramApi = urlParams.get('api');
  if (paramApi) {
    return paramApi.replace(/\/$/, "");
  }
  if (window.CAREBRIDGE_API_URL && typeof window.CAREBRIDGE_API_URL === 'string') {
    return window.CAREBRIDGE_API_URL.replace(/\/$/, "");
  }
  return "";
};

// Voice States
const State = {
  IDLE: 'IDLE',
  LISTENING: 'LISTENING',
  THINKING: 'THINKING',
  SPEAKING: 'SPEAKING',
  ENDING: 'ENDING',
  ENDED: 'ENDED'
};

class TTSAdapter {
  constructor() {
    this.synth = window.speechSynthesis;
    this.isMuted = false;
    this.currentUtterance = null;
    this.currentAudio = null;
    this.audioElement = null;
    this.voice = null;
    this.lang = 'en-US';
    this._initVoice();
  }

  unlockAudio() {
    if (!this.audioElement) {
      this.audioElement = new Audio();
    }
    try {
      this.audioElement.src = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=";
      const p = this.audioElement.play();
      if (p !== undefined) {
        p.then(() => {
          this.audioElement.pause();
          this.audioElement.currentTime = 0;
        }).catch(() => {});
      }
    } catch (e) {}
  }

  setLanguage(lang) {
    this.lang = lang;
    this._initVoice();
  }

  _initVoice() {
    if (!this.synth) return;
    const loadVoices = () => {
      const voices = this.synth.getVoices();
      if (this.lang.startsWith('hi')) {
        this.voice = voices.find(v => v.lang.startsWith('hi') && (v.name.includes('Google') || v.name.includes('Natural') || v.name.includes('Swara') || v.name.includes('Madhur') || v.name.includes('Lekha')))
                     || voices.find(v => v.lang.startsWith('hi'))
                     || null;
      } else {
        this.voice = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Samantha') || v.name.includes('Zira') || v.name.includes('Heera') || v.name.includes('Ravi'))) 
                     || voices.find(v => v.lang.startsWith('en')) 
                     || null;
      }
    };
    loadVoices();
    if (this.synth.onvoiceschanged !== undefined) {
      this.synth.onvoiceschanged = loadVoices;
    }
  }

  playAudioBase64(audioBase64, onStart, onEnd, onError) {
    if (this.isMuted) {
      if (onStart) onStart();
      setTimeout(() => { if (onEnd) onEnd(); }, 500);
      return;
    }

    this.stop();

    try {
      if (!this.audioElement) {
        this.audioElement = new Audio();
      }
      this.currentAudio = this.audioElement;
      this.audioElement.src = "data:audio/wav;base64," + audioBase64;

      this.audioElement.onplay = () => {
        if (onStart) onStart();
      };

      this.audioElement.onended = () => {
        this.currentAudio = null;
        if (onEnd) onEnd();
      };

      this.audioElement.onerror = (e) => {
        console.warn("Audio playback error, falling back to speech synthesis:", e);
        this.currentAudio = null;
        if (onError) onError(e);
        else if (onEnd) onEnd();
      };

      const playPromise = this.audioElement.play();
      if (playPromise !== undefined) {
        playPromise.catch((err) => {
          console.warn("Audio autoplay blocked or failed, falling back to speech synthesis:", err);
          this.currentAudio = null;
          if (onError) onError(err);
          else if (onEnd) onEnd();
        });
      }
    } catch (e) {
      console.warn("Audio creation failed:", e);
      if (onError) onError(e);
      else if (onEnd) onEnd();
    }
  }

  async speak(text, onStart, onEnd, onError, audioBase64 = null) {
    if (this.isMuted) {
      if (onStart) onStart();
      setTimeout(() => { if (onEnd) onEnd(); }, 500);
      return;
    }

    this.stop();

    const cleanText = text.replace(/[*_#`~]/g, '').trim();
    if (!cleanText) {
      if (onEnd) onEnd();
      return;
    }

    // 1. If high-fidelity voice audio is provided (e.g. Sarvam Bulbul for Hindi)
    if (audioBase64) {
      this.playAudioBase64(audioBase64, onStart, onEnd, (err) => {
        // Fallback to browser speech synthesis if audio playback fails
        this.speakWithSynth(cleanText, onStart, onEnd, onError);
      });
      return;
    }

    // 2. Otherwise direct browser native SpeechSynthesis (instant 0ms response for English)
    this.speakWithSynth(cleanText, onStart, onEnd, onError);
  }

  speakWithSynth(cleanText, onStart, onEnd, onError) {
    if (!this.synth) {
      if (onEnd) onEnd();
      return;
    }

    this.stop();

    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.lang = this.lang || 'en-US';
    utterance.rate = (this.lang && this.lang.startsWith('hi')) ? 0.90 : 0.95;
    utterance.pitch = 1.0;
    if (this.voice) utterance.voice = this.voice;

    utterance.onstart = () => { if (onStart) onStart(); };
    utterance.onend = () => {
      this.currentUtterance = null;
      if (onEnd) onEnd();
    };
    utterance.onerror = (e) => {
      this.currentUtterance = null;
      if (onError) onError(e);
      else if (onEnd) onEnd();
    };

    this.currentUtterance = utterance;
    this.synth.speak(utterance);
  }

  stop() {
    if (this.currentAudio) {
      try {
        this.currentAudio.pause();
        this.currentAudio.currentTime = 0;
      } catch (e) {}
      this.currentAudio = null;
    }
    if (this.synth) {
      try {
        this.synth.cancel();
      } catch (e) {}
    }
    this.currentUtterance = null;
  }

  toggleMute() {
    this.isMuted = !this.isMuted;
    if (this.isMuted) this.stop();
    return this.isMuted;
  }
}

class CareBridgeController {
  constructor() {
    this.currentState = State.IDLE;
    this.currentCallId = null;
    this.tts = new TTSAdapter();
    this.recognition = null;
    this.silenceTimer = null;
    this.speechPauseTimer = null;
    this.accumulatedTranscript = '';
    this.isCallActive = false;

    this.currentLang = 'en-US';
    this.btnLang = document.getElementById('btnLang');
    this.stateBadge = document.getElementById('stateBadge');
    this.stateText = document.getElementById('stateText');
    this.activeSpeechText = document.getElementById('activeSpeechText');
    this.btnStart = document.getElementById('btnStart');
    this.btnEnd = document.getElementById('btnEnd');
    this.btnMute = document.getElementById('btnMute');
    this.btnTheme = document.getElementById('btnTheme');
    this.transcriptList = document.getElementById('transcriptList');
    this.alertsContainer = document.getElementById('alertsContainer');
    this.statusIndicators = document.getElementById('statusIndicators');
    this.summaryModal = document.getElementById('summaryModal');
    this.errorBanner = document.getElementById('errorBanner');
    this.voiceOrb = document.getElementById('voiceOrb');
    this.voiceOrbWrapper = document.getElementById('voiceOrbWrapper');
    this.voiceOrbIcon = document.getElementById('voiceOrbIcon');

    this.useSarvamSTTOnly = false;
    this.audioStream = null;
    this.mediaRecorder = null;
    this.audioChunks = [];

    this._initBedsideClock();
    this._initMediaRecorder();
    this._initSpeechRecognition();
    this._bindEvents();
    this._checkServerWarmup();
  }

  _initBedsideClock() {
    const timeEl = document.getElementById('clockTime');
    const dateEl = document.getElementById('clockDate');
    if (!timeEl || !dateEl) return;

    const updateClock = () => {
      const now = new Date();
      timeEl.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      dateEl.textContent = now.toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric' });
    };

    updateClock();
    setInterval(updateClock, 1000);
  }

  async _initMediaRecorder() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      try {
        this.audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      } catch (err) {
        console.warn("Microphone access for Sarvam STT:", err);
      }
    }
  }

  _startMediaRecording() {
    if (!this.audioStream) return;
    try {
      this.audioChunks = [];
      const mime = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : (MediaRecorder.isTypeSupported('audio/mp4') ? 'audio/mp4' : 'audio/wav');
      this.mediaRecorder = new MediaRecorder(this.audioStream, { mimeType: mime });
      this.mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) this.audioChunks.push(e.data);
      };
      this.mediaRecorder.start(250);
    } catch (e) {
      console.warn("MediaRecorder start failed:", e);
    }
  }

  async _stopMediaRecordingAndTranscribe() {
    if (!this.mediaRecorder || this.mediaRecorder.state === 'inactive') return null;
    return new Promise((resolve) => {
      this.mediaRecorder.onstop = async () => {
        if (!this.audioChunks || this.audioChunks.length === 0) {
          resolve(null);
          return;
        }
        const mime = this.mediaRecorder.mimeType || 'audio/webm';
        const blob = new Blob(this.audioChunks, { type: mime });
        this.audioChunks = [];
        try {
          const apiBase = getApiBase();
          const formData = new FormData();
          formData.append('file', blob, 'user_voice.' + (mime.includes('webm') ? 'webm' : 'wav'));
          formData.append('language', this.currentLang && this.currentLang.startsWith('hi') ? 'hi-IN' : 'en-IN');
          const resp = await fetch(`${apiBase}/api/voice/stt`, {
            method: 'POST',
            body: formData
          });
          if (resp.ok) {
            const data = await resp.json();
            resolve(data.transcript || null);
            return;
          }
        } catch (err) {
          console.warn("Sarvam STT failed:", err);
        }
        resolve(null);
      };
      try {
        this.mediaRecorder.stop();
      } catch (e) {
        resolve(null);
      }
    });
  }

  _initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.info("Native Web Speech Recognition not available in this browser; engaging Sarvam AI Saaras STT.");
      this.useSarvamSTTOnly = true;
      return;
    }

    this.recognition = new SpeechRecognition();
    // Continuous recognition in WebKit Speech can occasionally drop or stall on silence
    // Setting continuous=false with instant auto-restart is universally robust across all Chrome/Edge builds
    this.recognition.continuous = false;
    this.recognition.interimResults = true;
    this.recognition.maxAlternatives = 1;
    this.recognition.lang = 'en-US';

    this.recognition.onstart = () => {
      this._resetSilenceTimer();
      if (this.currentState === State.LISTENING) {
        const isHi = this.currentLang && this.currentLang.startsWith('hi');
        this.activeSpeechText.textContent = isHi ? "आपकी आवाज़ सुन रहे हैं... (कृपया बोलिए)" : "Listening to you... (speak now)";
      }
    };

    this.recognition.onresult = (event) => {
      this.tts.stop();
      this._clearSilenceTimer();

      let interim = '';
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        const item = event.results[i];
        if (item.isFinal) {
          this.accumulatedTranscript += ' ' + item[0].transcript;
        } else {
          interim += item[0].transcript;
        }
      }

      const heardSoFar = (this.accumulatedTranscript + ' ' + interim).trim();
      if (heardSoFar) {
        this.activeSpeechText.textContent = `"${heardSoFar}"`;
      }

      if (this.speechPauseTimer) {
        clearTimeout(this.speechPauseTimer);
        this.speechPauseTimer = null;
      }

      // Natural 2000ms (2.0s) conversational pause interval - waits 2 seconds after speaker stops before taking speech
      if (heardSoFar) {
        this.speechPauseTimer = setTimeout(() => {
          if (this.isCallActive && this.currentState === State.LISTENING) {
            const captured = (this.accumulatedTranscript + ' ' + interim).trim();
            if (captured) {
              this.accumulatedTranscript = '';
              this.handleUserSpoke(captured);
            }
          }
        }, 2000);
      }
    };

    this.recognition.onerror = (event) => {
      console.warn("Speech Recognition notice:", event.error);
      if (event.error === 'not-allowed') {
        this.showError("Microphone permission was denied. Please allow microphone access in your browser address bar.");
        this.transitionTo(State.IDLE);
        this.isCallActive = false;
        this.updateButtons();
      } else if (event.error === 'no-speech') {
        // Natural silence, silenceTimer will handle gently
      } else {
        if (this.isCallActive && this.currentState === State.LISTENING) {
          setTimeout(() => this.startListening(), 400);
        }
      }
    };

    this.recognition.onend = () => {
      // If recognition stopped but we are still listening, restart immediately
      if (this.isCallActive && this.currentState === State.LISTENING) {
        setTimeout(() => {
          if (this.isCallActive && this.currentState === State.LISTENING) {
            try {
              this.recognition.start();
            } catch (e) {
              // Already started or busy
            }
          }
        }, 150);
      }
    };
  }

  _bindEvents() {
    this.btnStart.addEventListener('click', () => this.startCall());
    this.btnEnd.addEventListener('click', () => this.endCall());
    if (this.voiceOrb) {
      this.voiceOrb.addEventListener('click', () => {
        if (!this.isCallActive) {
          this.startCall();
        }
      });
    }

    this.btnMute.addEventListener('click', () => {
      const muted = this.tts.toggleMute();
      this.btnMute.innerHTML = muted ? '🔇 Unmute AI' : '🔊 AI Voice ON';
    });

    // Language toggle (English / Hindi)
    if (this.btnLang) {
      this.btnLang.addEventListener('click', () => this.toggleLanguage());
    }

    // Theme toggle (Warm Day / Ambient Night)
    if (this.btnTheme) {
      this.btnTheme.addEventListener('click', () => {
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        if (isDark) {
          document.documentElement.removeAttribute('data-theme');
          this.btnTheme.innerHTML = '☀️ Warm Day';
          localStorage.setItem('carebridge_theme', 'light');
        } else {
          document.documentElement.setAttribute('data-theme', 'dark');
          this.btnTheme.innerHTML = '🌙 Ambient Night';
          localStorage.setItem('carebridge_theme', 'dark');
        }
      });
      if (localStorage.getItem('carebridge_theme') === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
        this.btnTheme.innerHTML = '🌙 Ambient Night';
      }
    }

    // Text chat fallback form
    const textChatForm = document.getElementById('textChatForm');
    const manualInput = document.getElementById('manualInput');
    if (textChatForm && manualInput) {
      textChatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const text = manualInput.value.trim();
        if (!text) return;
        manualInput.value = '';
        if (!this.isCallActive) {
          // If call is not started yet, start it first then handle
          this.startCall().then(() => {
            setTimeout(() => this.handleUserSpoke(text), 1500);
          });
        } else {
          this.handleUserSpoke(text);
        }
      });
    }
  }

  toggleLanguage() {
    if (this.isCallActive) {
      this.showError("Please switch language before starting a conversation.");
      return;
    }
    if (this.currentLang === 'en-US') {
      this.currentLang = 'hi-IN';
      if (this.btnLang) this.btnLang.innerHTML = '🌐 हिन्दी';
      if (this.activeSpeechText) this.activeSpeechText.textContent = 'बातचीत शुरू करने के लिए Start Conversation पर क्लिक करें।';
      const manualInput = document.getElementById('manualInput');
      if (manualInput) manualInput.placeholder = 'या अपना जवाब यहाँ लिखें और Enter दबाएं...';
    } else {
      this.currentLang = 'en-US';
      if (this.btnLang) this.btnLang.innerHTML = '🌐 English';
      if (this.activeSpeechText) this.activeSpeechText.textContent = 'Press Start Conversation to begin.';
      const manualInput = document.getElementById('manualInput');
      if (manualInput) manualInput.placeholder = 'Or type your reply here and press Enter...';
    }
    this.tts.setLanguage(this.currentLang);
    if (this.recognition) {
      this.recognition.lang = this.currentLang;
    }
  }

  _clearSilenceTimer() {
    if (this.silenceTimer) {
      clearTimeout(this.silenceTimer);
      this.silenceTimer = null;
    }
  }

  _resetSilenceTimer() {
    this._clearSilenceTimer();
    // Wait a generous 15 seconds of total silence before any gentle check-in prompt
    this.silenceTimer = setTimeout(() => {
      if (this.isCallActive && this.currentState === State.LISTENING && !this.accumulatedTranscript.trim()) {
        const isHi = this.currentLang && this.currentLang.startsWith('hi');
        const prompt = isHi
          ? "मैं आपकी बात सुन रहा हूँ। आप आराम से बोल सकते हैं।"
          : "I am still here listening. Please take your time.";
        this.tts.speak(
          prompt,
          () => this.transitionTo(State.SPEAKING),
          () => {
            if (this.isCallActive) this.startListening();
          }
        );
      }
    }, 15000);
  }

  transitionTo(newState) {
    this.currentState = newState;
    this.stateBadge.className = `state-badge state-${newState}`;
    this.stateText.textContent = newState;

    if (this.voiceOrb) {
      this.voiceOrb.className = `voice-orb state-${newState}`;
    }
    if (this.voiceOrbWrapper) {
      this.voiceOrbWrapper.className = `voice-orb-wrapper state-${newState}`;
    }
    if (this.voiceOrbIcon) {
      if (newState === State.LISTENING) this.voiceOrbIcon.textContent = '👂';
      else if (newState === State.THINKING) this.voiceOrbIcon.textContent = '⏳';
      else if (newState === State.SPEAKING) this.voiceOrbIcon.textContent = '🔊';
      else this.voiceOrbIcon.textContent = '🎙️';
    }

    const isHi = this.currentLang && this.currentLang.startsWith('hi');
    if (newState === State.LISTENING) {
      this.tts.stop(); // Absolute silence while listening to elder
      this.activeSpeechText.textContent = isHi ? "आपकी आवाज़ सुन रहे हैं... (कृपया बोलिए)" : "Listening to you... (please speak)";
    } else if (newState === State.THINKING) {
      this.tts.stop(); // Absolute silence while backend processes LLM turn
      this.activeSpeechText.textContent = isHi ? "CareBridge सोच रहा है..." : "CareBridge is thinking...";
    } else if (newState === State.SPEAKING) {
      // Will be populated with the active question text directly
    } else if (newState === State.IDLE) {
      this.tts.stop();
      this.activeSpeechText.textContent = isHi ? "बातचीत शुरू करने के लिए Start Conversation पर क्लिक करें।" : "Press Start Conversation to begin.";
    } else if (newState === State.ENDING) {
      this.tts.stop();
      this.activeSpeechText.textContent = isHi ? "कॉल समाप्त हो रही है, रिपोर्ट तैयार की जा रही है..." : "Concluding call and preparing summary...";
    } else if (newState === State.ENDED) {
      this.tts.stop();
      this.activeSpeechText.textContent = isHi ? "बातचीत समाप्त हो गई है।" : "Conversation ended.";
    }

    this.updateButtons();
  }

  updateButtons() {
    this.btnStart.disabled = this.isCallActive;
    this.btnEnd.disabled = !this.isCallActive;
  }

  showError(msg) {
    this.errorBanner.textContent = msg;
    this.errorBanner.style.display = 'block';
    setTimeout(() => {
      this.errorBanner.style.display = 'none';
    }, 8000);
  }

  async _checkServerWarmup() {
    const apiBase = getApiBase();
    if (!apiBase) return; // Same origin, no external warmup needed
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3500);
      const resp = await fetch(`${apiBase}/api/health`, { signal: controller.signal });
      clearTimeout(timeoutId);
      if (resp.ok) {
        console.log("CareBridge backend is awake and healthy.");
      }
    } catch (e) {
      console.log("CareBridge backend may be sleeping (cold start). Initiating warm-up ping...");
      if (this.activeSpeechText && this.currentState === State.IDLE) {
        this.activeSpeechText.textContent = "Connecting to server (waking up, please give it a moment)...";
      }
      const warmInterval = setInterval(async () => {
        try {
          const check = await fetch(`${apiBase}/api/health`);
          if (check.ok) {
            clearInterval(warmInterval);
            if (this.activeSpeechText && this.currentState === State.IDLE) {
              this.activeSpeechText.textContent = "Press Start Conversation to begin.";
            }
          }
        } catch (err) {}
      }, 5000);
      setTimeout(() => clearInterval(warmInterval), 60000);
    }
  }

  async startCall() {
    try {
      this.tts.unlockAudio();
      this.hideSummary();
      this.errorBanner.style.display = 'none';
      this.isCallActive = true;
      this.transcriptList.innerHTML = '';
      this.alertsContainer.innerHTML = '<p style="color: #64748b; font-size: 15px;">No active safety alerts.</p>';
      this.transitionTo(State.THINKING);
      this.activeSpeechText.textContent = "Connecting to server (waking up if sleeping, please wait)...";

      // Determine backend origin
      const apiBase = getApiBase();
      const clientHour = new Date().getHours();
      const resp = await fetch(`${apiBase}/api/session/start`, { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          language: this.currentLang.startsWith('hi') ? 'hi' : 'en',
          clientHour: clientHour
        })
      });
      if (!resp.ok) throw new Error("Could not start session.");
      const data = await resp.json();

      this.currentCallId = data.callId;
      this.renderStatus(data.session.elderStatus);
      this.renderAlerts(data.session.activeConcerns || []);

      // AI initial greeting
      const greeting = data.greeting;
      this.appendTranscript('assistant', greeting);

      // Speak greeting then auto-start listening with calm 750ms interval
      this.tts.speak(
        greeting,
        () => {
          this.transitionTo(State.SPEAKING);
          this.activeSpeechText.textContent = `"${greeting}"`;
        },
        () => {
          if (this.isCallActive) {
            setTimeout(() => {
              if (this.isCallActive && this.currentState !== State.ENDED) {
                this.startListening();
              }
            }, 750);
          }
        },
        () => {
          if (this.isCallActive) {
            setTimeout(() => {
              if (this.isCallActive && this.currentState !== State.ENDED) {
                this.startListening();
              }
            }, 750);
          }
        },
        data.audio_base64
      );
    } catch (err) {
      console.error(err);
      this.showError("Failed to connect to backend server. Make sure the server is running.");
      this.isCallActive = false;
      this.transitionTo(State.IDLE);
    }
  }

  async startListening() {
    if (!this.isCallActive) return;
    this.accumulatedTranscript = '';
    if (this.speechPauseTimer) {
      clearTimeout(this.speechPauseTimer);
      this.speechPauseTimer = null;
    }
    this.transitionTo(State.LISTENING);

    if (this.useSarvamSTTOnly) {
      if (!this.audioStream) await this._initMediaRecorder();
      this.activeSpeechText.textContent = "Listening to you with Sarvam AI STT... (speak now)";
      this._startMediaRecording();
      this._resetSilenceTimer();
      return;
    }

    if (this.recognition) {
      try {
        this.recognition.stop();
      } catch (e) {}
      setTimeout(() => {
        if (this.isCallActive && this.currentState === State.LISTENING) {
          try {
            this.recognition.start();
          } catch (e) {
            // In case it was already active
          }
        }
      }, 100);
    }
  }

  restartListening() {
    if (this.isCallActive && this.currentState === State.LISTENING) {
      this.startListening();
    }
  }

  stopListening() {
    this._clearSilenceTimer();
    if (this.speechPauseTimer) {
      clearTimeout(this.speechPauseTimer);
      this.speechPauseTimer = null;
    }
    if (this.useSarvamSTTOnly) {
      this._stopMediaRecordingAndTranscribe().then((text) => {
        if (text && text.trim() && this.isCallActive) {
          this.handleUserSpoke(text.trim());
        }
      });
    }
    if (this.recognition) {
      try { this.recognition.stop(); } catch (e) {}
    }
  }

  async handleUserSpoke(userText) {
    this.stopListening();
    this.transitionTo(State.THINKING);
    this.appendTranscript('user', userText);

    try {
      const apiBase = getApiBase();
      const resp = await fetch(`${apiBase}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          callId: this.currentCallId,
          message: userText
        })
      });

      if (!resp.ok) {
        throw new Error("Chat request failed");
      }

      const data = await resp.json();

      // Update Live UI
      this.renderStatus(data.elderStatus);
      this.renderAlerts(data.activeConcerns);

      const aiReply = data.reply;
      this.appendTranscript('assistant', aiReply);

      // Speak AI reply -> on end, wait calm 750ms before listening again or conclude call if call_ended
      this.tts.speak(
        aiReply,
        () => {
          this.transitionTo(State.SPEAKING);
          this.activeSpeechText.textContent = `"${aiReply}"`;
        },
        () => {
          if (data.call_ended) {
            console.log("CareBridge wrap-up reached. Automatically ending call and preparing summary...");
            this.endCall();
          } else if (this.isCallActive) {
            setTimeout(() => {
              if (this.isCallActive && this.currentState !== State.ENDED) {
                this.startListening();
              }
            }, 750);
          }
        },
        () => {
          if (data.call_ended) {
            this.endCall();
          } else if (this.isCallActive) {
            setTimeout(() => {
              if (this.isCallActive && this.currentState !== State.ENDED) {
                this.startListening();
              }
            }, 750);
          }
        },
        data.audio_base64
      );

    } catch (err) {
      console.error("Chat error:", err);
      // Recover gracefully - never leave stuck in THINKING
      const isHi = this.currentLang && this.currentLang.startsWith('hi');
      const fallbackMsg = isHi
        ? "माफ़ कीजिए, मुझे सुनने में थोड़ी रुकावट आई। क्या आप इसे दोबारा दोहरा सकते हैं?"
        : "I'm sorry, I had a brief moment of static. Could you say that again?";
      this.appendTranscript('assistant', fallbackMsg);
      this.tts.speak(
        fallbackMsg,
        () => this.transitionTo(State.SPEAKING),
        () => {
          if (this.isCallActive) this.startListening();
        }
      );
    }
  }

  async endCall() {
    if (!this.isCallActive) return;
    this.isCallActive = false;
    this.stopListening();
    this.tts.stop();
    this.transitionTo(State.ENDING);

    try {
      const apiBase = getApiBase();
      const resp = await fetch(`${apiBase}/api/session/end`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ callId: this.currentCallId })
      });

      if (!resp.ok) throw new Error("Failed to end session.");
      const data = await resp.json();

      this.transitionTo(State.ENDED);
      this.displaySummary(data.report);

    } catch (err) {
      console.error("End call error:", err);
      this.transitionTo(State.ENDED);
      this.showError("Could not retrieve summary report from server.");
    }
  }

  appendTranscript(role, text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    const sender = document.createElement('div');
    sender.className = 'msg-sender';
    sender.textContent = role === 'user' ? 'Elder' : 'CareBridge';

    const content = document.createElement('div');
    content.textContent = text;

    msgDiv.appendChild(sender);
    msgDiv.appendChild(content);
    this.transcriptList.appendChild(msgDiv);
    this.transcriptList.scrollTop = this.transcriptList.scrollHeight;
  }

  renderStatus(status) {
    if (!status) return;
    this.statusIndicators.innerHTML = '';
    const displayKeys = [
      { key: 'mood', label: 'Mood' },
      { key: 'sleep', label: 'Sleep' },
      { key: 'breakfast', label: 'Breakfast' },
      { key: 'lunch', label: 'Lunch' },
      { key: 'dinner', label: 'Dinner' },
      { key: 'medicationTaken', label: 'Medication' },
      { key: 'painPresent', label: 'Pain' },
      { key: 'painLocation', label: 'Pain Area' },
      { key: 'dizziness', label: 'Dizziness' },
      { key: 'fallReported', label: 'Fall' }
    ];

    displayKeys.forEach(item => {
      let val = status[item.key];
      if (val === true) val = "Yes";
      else if (val === false) val = "No";
      else if (val === null || val === undefined) val = "—";

      const badge = document.createElement('div');
      badge.className = 'status-badge-item';
      badge.innerHTML = `
        <span class="status-key">${item.label}:</span>
        <span class="status-val">${val}</span>
      `;
      this.statusIndicators.appendChild(badge);
    });
  }

  renderAlerts(activeConcerns) {
    if (!this.alertsContainer) return;

    if (!activeConcerns || activeConcerns.length === 0) {
      this.alertsContainer.innerHTML = '<p class="empty-placeholder" style="padding: 8px 0;">No active safety alerts.</p>';
      return;
    }

    this.alertsContainer.innerHTML = '';
    activeConcerns.forEach(concern => {
      const alertDiv = document.createElement('div');
      alertDiv.className = `alert-item alert-${concern.level}`;
      const timeStr = concern.timestamp 
        ? new Date(concern.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) 
        : new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

      alertDiv.innerHTML = `
        <div class="alert-header">
          <span>Level: ${concern.level.toUpperCase()}</span>
          <span>${timeStr}</span>
        </div>
        <div><strong>Reason:</strong> ${concern.reason || 'Check required'}</div>
        <div><strong>Suggested Action:</strong> ${concern.action || 'Monitor elder'}</div>
      `;
      this.alertsContainer.appendChild(alertDiv);
    });
  }

  displaySummary(report) {
    if (!report) return;
    document.getElementById('sumOverallStatus').textContent = report.overall_status.replace('_', ' ');
    document.getElementById('sumOverallStatus').className = `overall-status-badge status-${report.overall_status}`;
    document.getElementById('sumOverview').textContent = report.summary;

    // Key points
    const kpList = document.getElementById('sumKeyPoints');
    kpList.innerHTML = '';
    (report.key_points || []).forEach(pt => {
      const li = document.createElement('li');
      li.textContent = pt;
      kpList.appendChild(li);
    });

    // Caregiver Actions
    const caList = document.getElementById('sumCaregiverActions');
    caList.innerHTML = '';
    (report.caregiver_actions || []).forEach(act => {
      const li = document.createElement('li');
      li.textContent = act;
      caList.appendChild(li);
    });

    // Concerns list
    const cList = document.getElementById('sumConcerns');
    cList.innerHTML = '';
    if (report.concerns && report.concerns.length > 0) {
      report.concerns.forEach(c => {
        const li = document.createElement('li');
        li.textContent = `[${c.level.toUpperCase()}] ${c.issue || c.reason || 'Concern noted'}`;
        cList.appendChild(li);
      });
    } else {
      cList.innerHTML = '<li>No major concerns detected during call.</li>';
    }

    this.summaryModal.style.display = 'block';
    this.summaryModal.scrollIntoView({ behavior: 'smooth' });
  }

  hideSummary() {
    this.summaryModal.style.display = 'none';
  }
}

// Instantiate on DOM load
window.addEventListener('DOMContentLoaded', () => {
  window.carebridge = new CareBridgeController();
});
