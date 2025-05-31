// Get references to all DOM elements
const serverFeedImg = document.getElementById('serverFeed');
const canvasOutput = document.getElementById('canvasOutput');
const ctx = canvasOutput.getContext('2d');

const toggleGlasses = document.getElementById('toggleGlasses');
const toggleMoustache = document.getElementById('toggleMoustache');
const toggleBwFilter = document.getElementById('toggleBwFilter');
const toggleSepiaFilter = document.getElementById('toggleSepiaFilter');
const toggleInvertFilter = document.getElementById('toggleInvertFilter');
const toggleVignette = document.getElementById('toggleVignette'); // New
const colorTintPicker = document.getElementById('colorTintPicker'); // New
const colorTintOpacity = document.getElementById('colorTintOpacity'); // New

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const downloadLink = document.getElementById('downloadLink');
const localRawAudioVideo = document.getElementById('localRawAudioVideo');

// Global variables for audio processing
let audioContext = null;
let microphoneSource = null;
let processedAudioStreamDestination = null;
let mediaRecorder = null;
let chunks = [];
let animationFrameId = null;
let videoStreamError = false; // To track if Flask video stream has errors

// --- Utility Functions ---

// Function to display temporary messages to the user (replaces alert/confirm)
function showMessage(message, type = 'info') {
    const messageBox = document.createElement('div');
    messageBox.className = `fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 text-white`;
    if (type === 'error') messageBox.classList.add('bg-red-800');
    else if (type === 'warning') messageBox.classList.add('bg-yellow-600');
    else messageBox.classList.add('bg-blue-600'); // Default info color

    messageBox.textContent = message;
    document.body.appendChild(messageBox);
    setTimeout(() => messageBox.remove(), 5000); // Remove after 5 seconds
}

// --- Audio Processing Setup ---
async function createAudioEffects() {
    // Initialize AudioContext only once
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }

    // Disconnect previous nodes if they exist to prevent multiple connections
    if (microphoneSource) microphoneSource.disconnect();

    // Create new audio nodes
    const waveShaperNode = audioContext.createWaveShaper();
    const compressorNode = audioContext.createDynamicsCompressor();
    const filterNodeHigh = audioContext.createBiquadFilter();
    const filterNodeLow = audioContext.createBiquadFilter();
    const delayNode = audioContext.createDelay(0.1);
    const feedbackGain = audioContext.createGain();
    const masterGainNode = audioContext.createGain();
    processedAudioStreamDestination = audioContext.createMediaStreamDestination();

    // WaveShaperNode (for autotune feel)
    // Adjusted 'k' for a less robotic, more subtle autotune effect
    const k = 20; // Original was 40. Experiment with values from 10 to 30.
    const n_samples = 44100;
    const curve = new Float32Array(n_samples);
    const deg = Math.PI / 180;
    for (let i = 0; i < n_samples; ++i) {
        const x = i * 2 / n_samples - 1;
        curve[i] = (3 + k) * x * 20 * deg / (Math.PI + k * Math.abs(x));
    }
    waveShaperNode.curve = curve;
    waveShaperNode.oversample = '4x';

    // DynamicsCompressorNode (More aggressive for smoother, consistent sound)
    compressorNode.threshold.setValueAtTime(-30, audioContext.currentTime);
    compressorNode.knee.setValueAtTime(30, audioContext.currentTime);
    compressorNode.ratio.setValueAtTime(15, audioContext.currentTime);
    compressorNode.attack.setValueAtTime(0.005, audioContext.currentTime);
    compressorNode.release.setValueAtTime(0.15, audioContext.currentTime);

    // BiquadFilterNode (High-Pass for noise reduction and clarity)
    filterNodeHigh.type = "highpass";
    filterNodeHigh.frequency.setValueAtTime(150, audioContext.currentTime);
    filterNodeHigh.Q.setValueAtTime(1, audioContext.currentTime);

    // BiquadFilterNode (Low-Pass for softness and removing harshness)
    filterNodeLow.type = "lowpass";
    filterNodeLow.frequency.setValueAtTime(5000, audioContext.currentTime);
    filterNodeLow.Q.setValueAtTime(1, audioContext.currentTime);

    // DelayNode (for a *very subtle* echo/reverb effect)
    delayNode.delayTime.setValueAtTime(0.01, audioContext.currentTime);
    feedbackGain.gain.setValueAtTime(0.2, audioContext.currentTime);

    // Master Gain Node (for overall volume control and perceived softness)
    masterGainNode.gain.setValueAtTime(0.8, audioContext.currentTime);

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        localRawAudioVideo.srcObject = stream;
        microphoneSource = audioContext.createMediaStreamSource(stream);

        // Connect the nodes in a chain
        microphoneSource.connect(waveShaperNode);
        waveShaperNode.connect(compressorNode);
        compressorNode.connect(filterNodeHigh);
        filterNodeHigh.connect(filterNodeLow);

        // Connect the main signal path to the delay
        filterNodeLow.connect(delayNode);

        // Create feedback loop: Delay output -> Gain -> Delay input
        delayNode.connect(feedbackGain);
        feedbackGain.connect(delayNode);

        // Mix the 'dry' (un-delayed) signal and the 'wet' (delayed) signal to the master gain node
        filterNodeLow.connect(masterGainNode);
        delayNode.connect(masterGainNode);

        // Connect the master gain node to the final destination
        masterGainNode.connect(processedAudioStreamDestination);

        console.log("Enhanced audio effects initialized.");
    } catch (err) {
        console.error("Error accessing microphone or setting up audio processing:", err);
        showMessage("Could not access microphone or set up audio processing. Recording may have no audio or use raw audio. Please grant permissions.", 'error');
    }
}

// --- Canvas Drawing Loop ---
function drawCanvasFrame() {
    ctx.save(); // Save the current state of the canvas context

    // Clear the entire canvas for the new frame
    ctx.clearRect(0, 0, canvasOutput.width, canvasOutput.height);

    // Ensure no client-side filters are applied here, Flask handles them.
    ctx.filter = 'none';

    // Draw the video frame from Flask stream
    if (serverFeedImg.complete && serverFeedImg.naturalWidth > 0 && !videoStreamError) {
        // Ensure canvas dimensions match video for correct drawing
        canvasOutput.width = serverFeedImg.naturalWidth;
        canvasOutput.height = serverFeedImg.naturalHeight;
        // Draw the image onto the canvas. It already has filters applied by Flask.
        ctx.drawImage(serverFeedImg, 0, 0, canvasOutput.width, canvasOutput.height);
    } else {
        // Display error message on canvas if video stream is not loading
        ctx.fillStyle = '#333'; // Dark background
        ctx.fillRect(0, 0, canvasOutput.width, canvasOutput.height);
        ctx.fillStyle = 'white';
        ctx.textAlign = 'center';
        ctx.font = '24px Inter';
        ctx.fillText("Camera Stream Unavailable", canvasOutput.width / 2, canvasOutput.height / 2 - 20);
        ctx.font = '16px Inter';
        ctx.fillText("Check Flask server console for camera errors.", canvasOutput.width / 2, canvasOutput.height / 2 + 10);
    }

    // --- Apply NEW Client-Side Filters (after drawing the Flask stream) ---

    // 1. Vignette Effect
    if (toggleVignette.checked) {
        const centerX = canvasOutput.width / 2;
        const centerY = canvasOutput.height / 2;
        const maxDim = Math.max(canvasOutput.width, canvasOutput.height);
        const radius = maxDim * 0.7; // Adjust for desired vignette size

        const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius);
        gradient.addColorStop(0, 'rgba(0,0,0,0)'); // Transparent in center
        gradient.addColorStop(0.7, 'rgba(0,0,0,0)'); // Still transparent
        gradient.addColorStop(1, 'rgba(0,0,0,0.7)'); // Opaque black at edges

        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvasOutput.width, canvasOutput.height);
    }

    // 2. Color Tint Overlay
    const tintOpacity = parseFloat(colorTintOpacity.value) / 100; // Convert range 0-100 to 0-1
    if (tintOpacity > 0) {
        ctx.globalAlpha = tintOpacity; // Set global transparency for the tint
        ctx.fillStyle = colorTintPicker.value; // Use the selected color
        ctx.fillRect(0, 0, canvasOutput.width, canvasOutput.height);
        ctx.globalAlpha = 1.0; // Reset globalAlpha for subsequent drawings
    }

    ctx.restore(); // Restore the canvas context to its state before ctx.save()

    // Request the next animation frame to continue the drawing loop
    animationFrameId = requestAnimationFrame(drawCanvasFrame);
}

// --- Video Stream Management ---
function updateVideoFeed() {
    const glassesOn = toggleGlasses.checked ? '1' : '0';
    const moustacheOn = toggleMoustache.checked ? '1' : '0';
    const bwOn = toggleBwFilter.checked ? '1' : '0';
    const sepiaOn = toggleSepiaFilter.checked ? '1' : '0';
    const invertOn = toggleInvertFilter.checked ? '1' : '0';

    // Construct the URL with all filter parameters
    // The `t` parameter forces the browser to reload the image, applying new filters.
    serverFeedImg.src = `/video_feed?glasses=${glassesOn}&moustache=${moustacheOn}&bw=${bwOn}&sepia=${sepiaOn}&invert=${invertOn}&t=${new Date().getTime()}`;
}

// --- Recording Logic ---
async function handleStartRecording() {
    if (audioContext && audioContext.state === 'suspended') {
        await audioContext.resume();
    }

    const canvasVideoStream = canvasOutput.captureStream(30); // 30 FPS from canvas

    let audioTrackToRecord;
    if (processedAudioStreamDestination && processedAudioStreamDestination.stream.getAudioTracks().length > 0) {
        audioTrackToRecord = processedAudioStreamDestination.stream.getAudioTracks()[0];
        console.log("Using processed (autotuned) audio track for recording.");
    } else if (localRawAudioVideo.srcObject && localRawAudioVideo.srcObject.getAudioTracks().length > 0) {
        audioTrackToRecord = localRawAudioVideo.srcObject.getAudioTracks()[0];
        console.warn("Processed audio not available, falling back to raw microphone audio for recording.");
        showMessage("Autotune effect not available for recording. Using raw microphone audio.", 'warning');
    } else {
        console.error("No audio track available for recording.");
        showMessage("Audio stream not available. Recording video only.", 'error');
    }

    const combinedStream = new MediaStream(canvasVideoStream.getVideoTracks());
    if (audioTrackToRecord) {
        combinedStream.addTrack(audioTrackToRecord);
    }

    try {
        const mimeTypes = [
            'video/webm; codecs="vp9, opus"',
            'video/webm; codecs="vp8, opus"',
            'video/webm; codecs="avc1, opus"',
            'video/mp4; codecs="avc1, mp4a.40.2"',
            'video/webm'
        ];
        let selectedMimeType = '';
        for (const mimeType of mimeTypes) {
            if (MediaRecorder.isTypeSupported(mimeType)) {
                selectedMimeType = mimeType;
                break;
            }
        }
        if (!selectedMimeType) {
            showMessage("No suitable MediaRecorder MIME type found for recording.", 'error');
            return;
        }
        console.log("Using MIME type for recording:", selectedMimeType);

        mediaRecorder = new MediaRecorder(combinedStream, { mimeType: selectedMimeType });
        chunks = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
                chunks.push(e.data);
            }
        };

        mediaRecorder.onstop = () => {
            if (chunks.length === 0) {
                console.warn("No data recorded.");
                downloadLink.style.display = 'none';
                showMessage("Recording failed or produced no data.", 'warning');
                return;
            }
            const blob = new Blob(chunks, { type: mediaRecorder.mimeType });
            downloadLink.href = URL.createObjectURL(blob);
            downloadLink.style.display = 'inline-block';
            chunks = [];
            startBtn.disabled = false;
            stopBtn.disabled = true;
        };

        mediaRecorder.start();
        startBtn.disabled = true;
        stopBtn.disabled = false;
        downloadLink.style.display = 'none';

    } catch (e) {
        console.error("MediaRecorder API is not supported or failed to initialize:", e);
        showMessage("Recording failed: MediaRecorder API not supported or error initializing.", 'error');
    }
}

function handleStopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }
}

// --- Initialization ---
function init() {
    // Event Listeners for UI controls
    toggleGlasses.addEventListener('change', updateVideoFeed);
    toggleMoustache.addEventListener('change', updateVideoFeed);

    // Flask-handled filters trigger updateVideoFeed
    toggleBwFilter.addEventListener('change', updateVideoFeed);
    toggleSepiaFilter.addEventListener('change', updateVideoFeed);
    toggleInvertFilter.addEventListener('change', updateVideoFeed);

    // New client-side filters trigger drawCanvasFrame directly
    toggleVignette.addEventListener('change', drawCanvasFrame);
    colorTintPicker.addEventListener('input', drawCanvasFrame);
    colorTintOpacity.addEventListener('input', drawCanvasFrame);

    startBtn.addEventListener('click', handleStartRecording);
    stopBtn.addEventListener('click', handleStopRecording);

    // Initial load of Flask video stream with default (off) filters
    serverFeedImg.src = `/video_feed?glasses=0&moustache=0&bw=0&sepia=0&invert=0&t=${new Date().getTime()}`;

    // Handle video stream loading and errors
    serverFeedImg.onload = () => {
        console.log("Video stream (serverFeedImg) loaded/reloaded.");
        // Ensure canvas dimensions match video for correct drawing
        canvasOutput.width = serverFeedImg.naturalWidth;
        canvasOutput.height = serverFeedImg.naturalHeight;
        videoStreamError = false; // Clear error if loaded successfully
        // Start canvas drawing loop
        if (!animationFrameId) { // Prevent multiple loops if already running
            drawCanvasFrame();
        }
    };

    serverFeedImg.onerror = () => {
        console.error("Error loading video stream from Flask backend. Please check Flask server console.");
        videoStreamError = true; // Set error state
        // Draw error message on canvas immediately
        drawCanvasFrame(); // Call to update canvas with error message
    };

    // Initialize audio effects
    createAudioEffects();

    // Cleanup on page unload (important for camera resources)
    window.addEventListener('beforeunload', () => {
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
        }
        if (audioContext) {
            audioContext.close();
        }
        if (microphoneSource) {
            microphoneSource.disconnect();
        }
        if (processedAudioStreamDestination) {
            processedAudioStreamDestination.disconnect();
        }
        // Flask backend handles camera release via atexit
    });
}

// Run init function when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', init);
