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
  if (window.location.port === "5500") {
    return "http://127.0.0.1:5500";
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
    this.voice = null;
    this.lang = 'en-US';
    this._initVoice();
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
        this.voice = voices.find(v => v.lang.startsWith('hi')) || null;
      } else {
        this.voice = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Samantha') || v.name.includes('Zira'))) 
                     || voices.find(v => v.lang.startsWith('en')) 
                     || null;
      }
    };
    loadVoices();
    if (this.synth.onvoiceschanged !== undefined) {
      this.synth.onvoiceschanged = loadVoices;
    }
  }

  speak(text, onStart, onEnd, onError) {
    if (this.isMuted) {
      if (onStart) onStart();
      setTimeout(() => { if (onEnd) onEnd(); }, 500);
      return;
    }

    if (!this.synth) {
      console.warn("SpeechSynthesis not supported.");
      if (onEnd) onEnd();
      return;
    }

    // Force cancel anything currently speaking or queued
    this.stop();

    const cleanText = text.replace(/[*_#`]/g, '').trim();
    if (!cleanText) {
      if (onEnd) onEnd();
      return;
    }

    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.lang = this.lang || 'en-US';
    utterance.rate = 0.95; // Slightly slower, calm cadence for elderly users
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
    if (this.synth) {
      this.synth.cancel();
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

    this._initSpeechRecognition();
    this._bindEvents();
  }

  _initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      this.showError("Speech Recognition is not supported in this browser. Please use Google Chrome or Microsoft Edge.");
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
        this.activeSpeechText.textContent = "Listening to you... (speak now)";
      }
    };

    this.recognition.onresult = (event) => {
      this.tts.stop();
      this._clearSilenceTimer();

      let interim = '';
      let finalTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; ++i) {
        const item = event.results[i];
        if (item.isFinal) {
          finalTranscript += item[0].transcript;
        } else {
          interim += item[0].transcript;
        }
      }

      const heardSoFar = (finalTranscript || interim).trim();
      if (heardSoFar) {
        this.activeSpeechText.textContent = `"${heardSoFar}"`;
      }

      if (finalTranscript.trim()) {
        const captured = finalTranscript.trim();
        this.handleUserSpoke(captured);
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
    this.btnMute.addEventListener('click', () => {
      const muted = this.tts.toggleMute();
      this.btnMute.innerHTML = muted ? '🔇 Unmute AI' : '🔊 Mute AI';
    });

    // Language toggle (English / Hindi)
    if (this.btnLang) {
      this.btnLang.addEventListener('click', () => this.toggleLanguage());
    }

    // Theme toggle
    if (this.btnTheme) {
      this.btnTheme.addEventListener('click', () => {
        const isLight = document.body.classList.toggle('light-theme');
        this.btnTheme.innerHTML = isLight ? '🌙 Dark Mode' : '☀️ Light Mode';
        localStorage.setItem('carebridge_theme', isLight ? 'light' : 'dark');
      });
      if (localStorage.getItem('carebridge_theme') === 'light') {
        document.body.classList.add('light-theme');
        this.btnTheme.innerHTML = '🌙 Dark Mode';
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
        this.tts.speak(
          "I am still here listening. Please take your time.",
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

    if (newState === State.LISTENING) {
      this.tts.stop(); // Absolute silence while listening to elder
      this.activeSpeechText.textContent = "Listening to you...";
    } else if (newState === State.THINKING) {
      this.tts.stop(); // Absolute silence while backend processes LLM turn
      this.activeSpeechText.textContent = "CareBridge is thinking...";
    } else if (newState === State.SPEAKING) {
      this.activeSpeechText.textContent = "CareBridge is speaking...";
    } else if (newState === State.IDLE) {
      this.tts.stop();
      this.activeSpeechText.textContent = "Press Start Conversation to begin.";
    } else if (newState === State.ENDING) {
      this.tts.stop();
      this.activeSpeechText.textContent = "Concluding call and preparing summary...";
    } else if (newState === State.ENDED) {
      this.tts.stop();
      this.activeSpeechText.textContent = "Conversation ended.";
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

  async startCall() {
    try {
      this.hideSummary();
      this.errorBanner.style.display = 'none';
      this.isCallActive = true;
      this.transcriptList.innerHTML = '';
      this.alertsContainer.innerHTML = '<p style="color: #64748b; font-size: 15px;">No active safety alerts.</p>';
      this.transitionTo(State.THINKING);

      // Determine backend origin
      const apiBase = getApiBase();
      const resp = await fetch(`${apiBase}/api/session/start`, { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ language: this.currentLang.startsWith('hi') ? 'hi' : 'en' })
      });
      if (!resp.ok) throw new Error("Could not start session.");
      const data = await resp.json();

      this.currentCallId = data.callId;
      this.renderStatus(data.session.elderStatus);
      this.renderAlerts(data.session.activeConcerns || []);

      // AI initial greeting
      const greeting = data.greeting;
      this.appendTranscript('assistant', greeting);

      // Speak greeting then auto-start listening
      this.tts.speak(
        greeting,
        () => this.transitionTo(State.SPEAKING),
        () => {
          if (this.isCallActive) {
            this.startListening();
          }
        },
        () => {
          if (this.isCallActive) this.startListening();
        }
      );
    } catch (err) {
      console.error(err);
      this.showError("Failed to connect to backend server. Make sure the server is running.");
      this.isCallActive = false;
      this.transitionTo(State.IDLE);
    }
  }

  startListening() {
    if (!this.isCallActive || !this.recognition) return;
    this.accumulatedTranscript = '';
    if (this.speechPauseTimer) {
      clearTimeout(this.speechPauseTimer);
      this.speechPauseTimer = null;
    }
    this.transitionTo(State.LISTENING);
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

      // Speak AI reply -> on end, automatically listen again
      this.tts.speak(
        aiReply,
        () => this.transitionTo(State.SPEAKING),
        () => {
          if (this.isCallActive) {
            this.startListening();
          }
        },
        () => {
          if (this.isCallActive) this.startListening();
        }
      );

    } catch (err) {
      console.error("Chat error:", err);
      // Recover gracefully - never leave stuck in THINKING
      const fallbackMsg = "I'm sorry, I had a brief moment of static. Could you say that again?";
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
      this.alertsContainer.innerHTML = '<p style="color: #64748b; font-size: 15px;">No active safety alerts.</p>';
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
