// Global State
let stream = null;
let captureInterval = null;
let isStreaming = false;
let isDemoMode = true;
let captureInFlight = false;
let pendingCapture = false;
let lastSendTime = Date.now();
let fpsIntervals = [];

// Prediction Smoothing State
let predictionHistory = [];
let stablePrediction = "";
let stableCounter = 0;
let sentenceWords = [];
let latestRecognitionState = {
    prediction: null,
    confidence: 0,
    top_predictions: [],
    sentence_buffer: [],
    recent_predictions: [],
    non_manual_features: {},
    attention: {},
    tts_text: ""
};
let assistantMode = 'translate';

// DOM Elements
const webcam = document.getElementById('webcam');
const overlayCanvas = document.getElementById('overlay-canvas');
const ctx = overlayCanvas.getContext('2d');
const placeholder = document.getElementById('placeholder');

const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const resetBtn = document.getElementById('reset-btn');
const clearBtn = document.getElementById('clear-btn');
const speakBtn = document.getElementById('speak-btn');
const saveBtn = document.getElementById('save-btn');
const ttsAutoCheckbox = document.getElementById('tts-auto-checkbox');
const sihDemoBtn = document.getElementById('sih-demo-btn');

const predSign = document.getElementById('pred-sign');
const predConf = document.getElementById('pred-conf');
const confProgress = document.getElementById('conf-progress');
const fpsDisplay = document.getElementById('fps-display');
const outputText = document.getElementById('output-text');
const logContent = document.getElementById('log-content');
const assistantResponse = document.getElementById('assistant-response');
const assistantConfidence = document.getElementById('assistant-confidence');
const assistantAlternatives = document.getElementById('assistant-alternatives');
const assistantCuesList = document.getElementById('assistant-cues-list');
const assistantInput = document.getElementById('assistant-input');
const assistantSendBtn = document.getElementById('assistant-send-btn');
const assistantExplainBtn = document.getElementById('assistant-explain-btn');
const assistantModeBtns = document.querySelectorAll('.assistant-mode-btn');

// Cues
const cueEbL = document.getElementById('cue-eb-l');
const cueEbR = document.getElementById('cue-eb-r');
const cueEyes = document.getElementById('cue-eyes');
const cueMouth = document.getElementById('cue-mouth');
const cueHead = document.getElementById('cue-head');

// Configuration
const FRAME_INTERVAL_MS = 100; // Capture at up to 10 FPS; requests are serialized below
const CONF_THRESHOLD = 0.60;
const STABILITY_THRESHOLD = 4;
const SMOOTHING_WINDOW_SIZE = 8;

// Event Listeners
startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);
resetBtn.addEventListener('click', resetBuffer);
clearBtn.addEventListener('click', clearText);
speakBtn.addEventListener('click', () => speakText(outputText.textContent));
saveBtn.addEventListener('click', saveText);
sihDemoBtn.addEventListener('click', toggleDemoMode);
assistantSendBtn.addEventListener('click', sendAssistantMessage);
assistantExplainBtn.addEventListener('click', () => requestAssistantResponse('explain', 'Explain this prediction'));
assistantInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
        sendAssistantMessage();
    }
});
assistantModeBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
        assistantMode = btn.dataset.mode;
        assistantModeBtns.forEach((item) => item.classList.remove('active'));
        btn.classList.add('active');
        requestAssistantResponse(assistantMode, '');
    });
});

// Initialize
logMessage("[System] Dashboard initialized. Waiting for user action.");
renderAssistantState();

// Toggle Judge Demo Mode
function toggleDemoMode() {
    isDemoMode = !isDemoMode;
    if (isDemoMode) {
        sihDemoBtn.classList.add('active');
        logMessage("[Demo] SIH Judge Demonstration Mode ENABLED. Overlay logs active.");
    } else {
        sihDemoBtn.classList.remove('active');
        logMessage("[Demo] SIH Demonstration Mode disabled.");
    }
}

// Log Messages to Diagnostics Terminal
function logMessage(msg) {
    const timestamp = new Date().toLocaleTimeString();
    logContent.innerHTML += `\n[${timestamp}] ${msg}`;
    logContent.scrollTop = logContent.scrollHeight;
}

// Start Video Capture
async function startCamera() {
    logMessage("[Webcam] Requesting camera access...");
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                frameRate: { ideal: 30, max: 30 }
            },
            audio: false
        });
        
        webcam.srcObject = stream;
        placeholder.style.display = 'none';
        isStreaming = true;
        
        // Update Buttons
        startBtn.disabled = true;
        stopBtn.disabled = false;
        
        await webcam.play();

        // Match overlays to the actual camera stream instead of assuming 640x480.
        overlayCanvas.width = webcam.videoWidth || 640;
        overlayCanvas.height = webcam.videoHeight || 480;
        
        // Start streaming frames
        captureInterval = setInterval(captureFrame, FRAME_INTERVAL_MS);
        logMessage("[Webcam] Stream started successfully. Sending frames...");
    } catch (err) {
        logMessage(`[Error] Camera access denied: ${err.message}`);
        alert("Camera access denied! Make sure you grant permissions and are not running another webcam app.");
    }
}

// Stop Video Capture
function stopCamera() {
    logMessage("[Webcam] Stopping stream...");
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    if (captureInterval) {
        clearInterval(captureInterval);
        captureInterval = null;
    }

    captureInFlight = false;
    pendingCapture = false;
    
    webcam.srcObject = null;
    placeholder.style.display = 'flex';
    isStreaming = false;
    
    // Clear canvas
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    
    // Update Buttons
    startBtn.disabled = false;
    stopBtn.disabled = true;
    
    // Reset prediction display
    predSign.textContent = "--";
    predConf.textContent = "0%";
    confProgress.style.width = "0%";
    
    // Reset badges
    resetBadges();
    latestRecognitionState = {
        prediction: null,
        confidence: 0,
        top_predictions: [],
        sentence_buffer: [],
        recent_predictions: [],
        non_manual_features: {},
        attention: {},
        tts_text: ""
    };
    renderAssistantState();
    
    logMessage("[Webcam] Stream stopped.");
}

// Reset Backend Sequence Buffer
async function resetBuffer() {
    logMessage("[Inference] Resetting backend buffer...");
    try {
        const response = await fetch('/reset', { method: 'POST' });
        if (response.ok) {
            predictionHistory = [];
            stablePrediction = "";
            stableCounter = 0;
            latestRecognitionState.recent_predictions = [];
            latestRecognitionState.sentence_buffer = [];
            latestRecognitionState.tts_text = "";
            renderAssistantState();
            logMessage("[Inference] Buffer reset completed successfully.");
        } else {
            logMessage("[Error] Failed to reset backend buffer.");
        }
    } catch (err) {
        logMessage(`[Error] Network error during buffer reset: ${err.message}`);
    }
}

// Reset UI badges
function resetBadges() {
    const badges = [cueEbL, cueEbR, cueEyes, cueMouth, cueHead];
    badges.forEach(b => {
        b.className = "cue-val badge-neutral";
        b.textContent = b.id === "cue-head" ? "CENTER" : (b.id === "cue-eyes" ? "OPEN" : (b.id === "cue-mouth" ? "CLOSED" : "NORMAL"));
    });
}

// Capture frame and send to server
function captureFrame() {
    if (!isStreaming) return;

    // Do not let slow CPU inference reorder requests. Keep one newest frame to
    // process immediately after the current request completes.
    if (captureInFlight) {
        pendingCapture = true;
        return;
    }

    captureInFlight = true;
    
    // Measure FPS
    const now = Date.now();
    const duration = now - lastSendTime;
    lastSendTime = now;
    fpsIntervals.push(1000 / duration);
    if (fpsIntervals.length > 10) fpsIntervals.shift();
    const currentFps = Math.round(fpsIntervals.reduce((a, b) => a + b, 0) / fpsIntervals.length);
    fpsDisplay.textContent = `${currentFps} FPS`;
    
    // Draw offscreen canvas to capture frame
    const captureCanvas = document.createElement('canvas');
    captureCanvas.width = webcam.videoWidth || 640;
    captureCanvas.height = webcam.videoHeight || 480;
    const captureCtx = captureCanvas.getContext('2d');
    
    // Keep the inference frame in the same orientation as the training videos.
    // The webcam preview may be mirrored by CSS, but the model must see raw order.
    captureCtx.drawImage(webcam, 0, 0, captureCanvas.width, captureCanvas.height);
    
    const base64Image = captureCanvas.toDataURL('image/jpeg', 0.82);
    
    // Send to backend
    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64Image })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === "error") {
            logMessage(`[Backend Error] ${data.message}`);
            return;
        }
        
        handleInferenceResult(data);
    })
    .catch(err => {
        logMessage(`[Error] Prediction API failed: ${err.message}`);
    })
    .finally(() => {
        captureInFlight = false;
        if (pendingCapture && isStreaming) {
            pendingCapture = false;
            captureFrame();
        }
    });
}

// Handle returned result
function handleInferenceResult(data) {
    const prediction = data.prediction;
    const confidence = data.confidence;
    const features = data.features;
    const topPredictions = data.top_predictions || [];
    const attention = data.attention || {};
    
    // 1. Update Diagnostics & Explainability Badges
    updateBadges(features);
    updateRecognitionState(prediction, confidence, topPredictions, features, attention);
    
    // 2. Draw visual overlays (Face, eyes, mouth boxes) on local canvas
    drawVisualOverlays(features);
    
    if (!prediction || prediction === "MODEL NOT TRAINED") {
        if (prediction === "MODEL NOT TRAINED") {
            predSign.textContent = "TRAIN ME";
            assistantResponse.textContent = "I'm waiting for a trained model. Add best_model.pth and try again.";
            logMessage("[Warn] Model weight file best_model.pth is missing inside models/ directory.");
        }
        return;
    }
    
    // 3. Smooth Predictions using Sliding Window Majority
    predictionHistory.push(prediction);
    if (predictionHistory.length > SMOOTHING_WINDOW_SIZE) {
        predictionHistory.shift();
    }
    
    // Find majority prediction in sliding window
    const counts = {};
    let majorityPred = "";
    let maxCount = 0;
    
    predictionHistory.forEach(pred => {
        counts[pred] = (counts[pred] || 0) + 1;
        if (counts[pred] > maxCount) {
            maxCount = counts[pred];
            majorityPred = pred;
        }
    });
    
    // Update live panel
    predSign.textContent = majorityPred;
    predConf.textContent = `${Math.round(confidence * 100)}%`;
    confProgress.style.width = `${confidence * 100}%`;
    renderAssistantState();
    
    // 4. Stability Logic to Commit Word to Output Sentence
    if (majorityPred === stablePrediction && confidence >= CONF_THRESHOLD) {
        stableCounter++;
        if (stableCounter === STABILITY_THRESHOLD) {
            commitWord(majorityPred);
        }
    } else {
        stablePrediction = majorityPred;
        stableCounter = 1;
    }
}

// Commit sign word/phrase to sentence
function commitWord(word) {
    // Avoid repeating the same word consecutively
    if (sentenceWords.length === 0 || sentenceWords[sentenceWords.length - 1] !== word) {
        sentenceWords.push(word);
        const fullText = sentenceWords.join(" ");
        outputText.textContent = fullText;
        latestRecognitionState.sentence_buffer = [...sentenceWords];
        latestRecognitionState.tts_text = fullText;
        renderAssistantState();
        logMessage(`[Translate] Committed Sign: "${word}"`);
        
        // Auto Text-to-Speech
        if (ttsAutoCheckbox.checked) {
            speakText(word);
        }
    }
}

function updateRecognitionState(prediction, confidence, topPredictions, features, attention) {
    latestRecognitionState.prediction = prediction;
    latestRecognitionState.confidence = confidence || 0;
    latestRecognitionState.top_predictions = topPredictions;
    latestRecognitionState.non_manual_features = features || {};
    latestRecognitionState.attention = attention || {};

    if (prediction) {
        const recent = latestRecognitionState.recent_predictions || [];
        if (recent[recent.length - 1] !== prediction) {
            recent.push(prediction);
            if (recent.length > 8) {
                recent.shift();
            }
        }
        latestRecognitionState.recent_predictions = recent;
    }
}

function renderAssistantState() {
    const features = latestRecognitionState.non_manual_features || {};
    const activeText = latestRecognitionState.sentence_buffer.length > 0
        ? latestRecognitionState.sentence_buffer.join(' ')
        : latestRecognitionState.prediction;

    assistantConfidence.textContent = `${Math.round((latestRecognitionState.confidence || 0) * 100)}%`;
    assistantAlternatives.textContent = formatTopAlternatives(latestRecognitionState.top_predictions || []);
    assistantCuesList.textContent = formatCueSummary(features);

    if (!activeText) {
        assistantResponse.textContent = "I'm waiting for an ISL sign. Start the camera and perform a sign.";
        return;
    }

    if (assistantMode === 'translate') {
        assistantResponse.textContent = `I detected: ${activeText.toUpperCase()}.`;
    } else if (assistantMode === 'speak') {
        assistantResponse.textContent = latestRecognitionState.tts_text
            ? `Ready to speak: ${latestRecognitionState.tts_text.toUpperCase()}.`
            : "I'm waiting for a recognized sentence to speak.";
    }
}

function formatTopAlternatives(topPredictions) {
    if (!topPredictions || topPredictions.length <= 1) {
        return "No alternatives yet.";
    }

    return topPredictions
        .slice(1)
        .map((item) => `${item.label.toUpperCase()} - ${Math.round(item.confidence * 100)}%`)
        .join(' | ');
}

function formatCueSummary(features) {
    return `Eyebrows: ${features.eyebrow_left || 'normal'} / ${features.eyebrow_right || 'normal'} | Eyes: ${features.eyes || 'open'} | Mouth: ${features.mouth || 'closed'} | Head: ${features.head || 'center'}`;
}

async function requestAssistantResponse(mode, message) {
    const payload = {
        mode,
        message,
        prediction: latestRecognitionState.prediction,
        confidence: latestRecognitionState.confidence,
        top_predictions: latestRecognitionState.top_predictions,
        sentence_buffer: latestRecognitionState.sentence_buffer,
        recent_predictions: latestRecognitionState.recent_predictions,
        non_manual_features: latestRecognitionState.non_manual_features,
        attention: latestRecognitionState.attention,
        tts_text: latestRecognitionState.tts_text || outputText.textContent
    };

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await response.json();
        assistantResponse.textContent = data.response || "I don't have enough information from the current recognition session to answer that.";
        assistantConfidence.textContent = `${Math.round((data.confidence || latestRecognitionState.confidence || 0) * 100)}%`;

        if (mode === 'speak' && data.speak_text) {
            speakText(data.speak_text);
        }
    } catch (err) {
        assistantResponse.textContent = "I don't have enough information from the current recognition session to answer that.";
        logMessage(`[Assistant Error] ${err.message}`);
    }
}

function sendAssistantMessage() {
    const message = assistantInput.value.trim();
    if (!message) {
        return;
    }

    requestAssistantResponse(assistantMode, message);
    assistantInput.value = '';
}

// Update badges UI
function updateBadges(cues) {
    if (!cues || !cues.confidence_cues) return;
    
    // Eyebrows
    updateBadgeElement(cueEbL, cues.eyebrow_left, cues.eyebrow_left !== "normal");
    updateBadgeElement(cueEbR, cues.eyebrow_right, cues.eyebrow_right !== "normal");
    
    // Eyes
    updateBadgeElement(cueEyes, cues.eyes, cues.eyes !== "open");
    
    // Mouth
    updateBadgeElement(cueMouth, cues.mouth, cues.mouth !== "closed");
    
    // Head
    updateBadgeElement(cueHead, cues.head, cues.head !== "center");
}

function updateBadgeElement(el, val, isInteresting) {
    el.textContent = val;
    if (isInteresting) {
        el.className = "cue-val badge-active";
    } else {
        el.className = "cue-val badge-neutral";
    }
}

// Draw visual overlay guides for SIH Presentation Mode
function drawVisualOverlays(cues) {
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    
    if (!cues || !cues.confidence_cues) return;
    
    // Mock locations for visual feedback overlay (centered on canvas)
    const centerX = overlayCanvas.width / 2;
    const centerY = overlayCanvas.height / 2 - 20;
    
    // Draw Face Bounding Box
    ctx.strokeStyle = "rgba(0, 188, 212, 0.4)";
    ctx.lineWidth = 2;
    ctx.strokeRect(centerX - 100, centerY - 120, 200, 240);
    
    // Bounding Box title
    ctx.fillStyle = "rgba(0, 188, 212, 0.8)";
    ctx.font = "11px Courier New";
    ctx.fillText("AI_FACE_TRACKER_ACTIVE", centerX - 95, centerY - 100);
    
    // Estimate roll tilt angle for drawing
    const roll = cues.confidence_cues.roll || 0.0;
    const yaw = cues.confidence_cues.yaw || 0.0;
    
    // Draw Eyes trackers
    ctx.fillStyle = cues.eyes.includes("blink") ? "#f44336" : "#00bcd4";
    ctx.beginPath();
    ctx.arc(centerX - 40 - (yaw*0.3), centerY - 30, 8, 0, 2 * Math.PI); // Left Eye
    ctx.arc(centerX + 40 - (yaw*0.3), centerY - 30, 8, 0, 2 * Math.PI); // Right Eye
    ctx.fill();
    
    // Draw Eyebrows trackers
    ctx.strokeStyle = cues.eyebrow_left !== "normal" ? "#00bcd4" : "#9fa8da";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(centerX - 60, centerY - 45 - (cues.eyebrow_left === "raised" ? 5 : 0));
    ctx.lineTo(centerX - 25, centerY - 45 - (cues.eyebrow_left === "raised" ? 5 : 0));
    ctx.stroke();
    
    ctx.strokeStyle = cues.eyebrow_right !== "normal" ? "#00bcd4" : "#9fa8da";
    ctx.beginPath();
    ctx.moveTo(centerX + 25, centerY - 45 - (cues.eyebrow_right === "raised" ? 5 : 0));
    ctx.lineTo(centerX + 60, centerY - 45 - (cues.eyebrow_right === "raised" ? 5 : 0));
    ctx.stroke();
    
    // Draw Mouth tracker
    ctx.strokeStyle = cues.mouth !== "closed" ? "#4caf50" : "#607d8b";
    ctx.lineWidth = 3;
    ctx.beginPath();
    if (cues.mouth === "wide open") {
        ctx.arc(centerX, centerY + 40, 18, 0, 2 * Math.PI);
    } else if (cues.mouth === "open") {
        ctx.ellipse(centerX, centerY + 40, 20, 8, 0, 0, 2 * Math.PI);
    } else {
        // Closed line
        ctx.moveTo(centerX - 25, centerY + 40);
        ctx.lineTo(centerX + 25, centerY + 40);
    }
    ctx.stroke();
    
    // Draw head orientation visual cue vector
    if (isDemoMode) {
        ctx.strokeStyle = "#4caf50";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        // Draw vector line based on yaw and pitch
        const vectorEndX = centerX - (yaw * 2);
        const vectorEndY = centerY + (cues.confidence_cues.pitch * 2);
        ctx.lineTo(vectorEndX, vectorEndY);
        ctx.stroke();
        
        ctx.fillStyle = "#4caf50";
        ctx.beginPath();
        ctx.arc(vectorEndX, vectorEndY, 4, 0, 2 * Math.PI);
        ctx.fill();
        
        // Print numeric metrics inside overlay for judges
        ctx.fillStyle = "rgba(255,255,255,0.7)";
        ctx.font = "10px monospace";
        ctx.fillText(`YAW: ${Math.round(yaw)}°`, centerX - 90, centerY + 90);
        ctx.fillText(`PITCH: ${Math.round(cues.confidence_cues.pitch)}°`, centerX - 90, centerY + 105);
        ctx.fillText(`ROLL: ${Math.round(roll)}°`, centerX - 90, centerY + 120);
    }
}

// Clear translation output
function clearText() {
    sentenceWords = [];
    outputText.textContent = "Waiting for recognized signs...";
    latestRecognitionState.sentence_buffer = [];
    latestRecognitionState.tts_text = "";
    renderAssistantState();
    logMessage("[System] Output text cleared.");
    resetBuffer();
}

// Text-to-Speech (TTS)
function speakText(text) {
    if (!text || text.includes("Waiting for recognized signs")) return;
    
    if ('speechSynthesis' in window) {
        // Cancel ongoing speak
        window.speechSynthesis.cancel();
        
        const utterance = new SpeechSynthesisUtterance(text);
        const voices = window.speechSynthesis.getVoices();
        let selectedVoice = null;
        
        if (voices && voices.length > 0) {
            // Find best voice match: en-IN, then en-US, then any English voice
            selectedVoice = voices.find(v => v.lang === 'en-IN' || v.lang === 'en_IN') || 
                            voices.find(v => v.lang === 'en-US' || v.lang === 'en_US') || 
                            voices.find(v => v.lang.startsWith('en'));
        }
        
        if (selectedVoice) {
            utterance.voice = selectedVoice;
            utterance.lang = selectedVoice.lang;
        } else {
            utterance.lang = 'en-US'; // Standard fallback
        }
        
        utterance.rate = 0.9;
        window.speechSynthesis.speak(utterance);
        logMessage(`[TTS] Spoke: "${text}"`);
    } else {
        logMessage("[TTS Error] Web Speech API not supported in this browser.");
    }
}

// Save recognized text to local file
function saveText() {
    const text = outputText.textContent;
    if (!text || text.includes("Waiting for recognized signs")) return;
    
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'GESCOM_Translation.txt';
    link.click();
    logMessage("[System] Translation saved as GESCOM_Translation.txt");
}
