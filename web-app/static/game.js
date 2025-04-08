// === Invincible Flappy Game (Voice Controlled with Custom CNN Model) ===
const canvas = document.querySelector("canvas");
const ctx = canvas.getContext("2d");
canvas.width = 800;
canvas.height = 400;

let player = { x: 50, y: 150, width: 20, height: 20, vy: 0 };
let gravity = 0.005;
let jump = -0.5;
let fall = 0.5;
let pipes = [];
let score = 0;
let moving = true;
let gameOver = false;
let isRecording = false;
let audioContext;
let mediaRecorder;
let audioChunks = [];
let recordingInterval;
let statusElement;
let commandLog;

const pipeGap = 220;
const pipeWidth = 30;
let pipeInterval = 150;
let frameCount = 0;

// === Socket.io connection ===
const socket = io();

// Initialize status display
function initializeUI() {
  // Create status element if it doesn't exist
  if (!statusElement) {
    statusElement = document.createElement('div');
    statusElement.style.position = 'absolute';
    statusElement.style.bottom = '10px';
    statusElement.style.left = '10px';
    statusElement.style.color = 'white';
    statusElement.style.background = 'rgba(0,0,0,0.5)';
    statusElement.style.padding = '5px 10px';
    statusElement.style.borderRadius = '5px';
    document.body.appendChild(statusElement);
  }
  
  // Create command log if it doesn't exist
  if (!commandLog) {
    commandLog = document.createElement('div');
    commandLog.style.position = 'absolute';
    commandLog.style.top = '60px';
    commandLog.style.right = '10px';
    commandLog.style.color = 'white';
    commandLog.style.background = 'rgba(0,0,0,0.5)';
    commandLog.style.padding = '5px 10px';
    commandLog.style.borderRadius = '5px';
    commandLog.style.maxHeight = '200px';
    commandLog.style.overflow = 'auto';
    document.body.appendChild(commandLog);
  }
}

// Update status
function updateStatus(text) {
  if (statusElement) {
    statusElement.textContent = text;
  }
}

// Log command
function logCommand(command) {
  if (commandLog) {
    const entry = document.createElement('div');
    entry.textContent = `${new Date().toLocaleTimeString()}: ${command}`;
    commandLog.prepend(entry);
    
    // Keep only last 10 commands
    while (commandLog.children.length > 10) {
      commandLog.removeChild(commandLog.lastChild);
    }
  }
}

// === Draw text ===
function drawText(text, x, y, color = "white", size = 24) {
  ctx.fillStyle = color;
  ctx.font = `${size}px Arial`;
  ctx.fillText(text, x, y);
}

// === Generate pipes ===
function createPipe() {
  let topHeight = Math.floor(Math.random() * (canvas.height - pipeGap - 60)) + 20;
  pipes.push({ x: canvas.width, top: topHeight, passed: false });
}

// === Game reset ===
function resetGame() {
  player.y = 150;
  player.vy = 0;
  pipes = [];
  score = 0;
  moving = true;
  gameOver = false;
}

// === Update game state ===
function update() {
  if (moving) {
    player.vy += gravity;
    player.y += player.vy;

    // Keep player in bounds (top)
    if (player.y < 0) {
      player.y = 0;
      player.vy = 0;
    }
  }

  // Create pipes
  if (frameCount % pipeInterval === 0) createPipe();

  // Move pipes
  pipes.forEach(pipe => pipe.x -= 3);

  // Remove off-screen pipes
  pipes = pipes.filter(pipe => pipe.x + pipeWidth > 0);

  // Score logic
  pipes.forEach(pipe => {
    if (pipe.x + pipeWidth < player.x && !pipe.passed) {
      pipe.passed = true;
      score++;
    }
  });

  // Game over logic (fall below screen)
  if (player.y > canvas.height) {
    gameOver = true;
    moving = false;
  }

  frameCount++;
}

// === Render everything ===
function render() {
  ctx.fillStyle = "#aaa";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Draw player
  ctx.fillStyle = "blue";
  ctx.fillRect(player.x, player.y, player.width, player.height);

  // Draw pipes
  ctx.fillStyle = "green";
  pipes.forEach(pipe => {
    ctx.fillRect(pipe.x, 0, pipeWidth, pipe.top);
    ctx.fillRect(pipe.x, pipe.top + pipeGap, pipeWidth, canvas.height);
  });

  // Draw score
  drawText(`Score: ${score}`, 10, 30);
  
  // Draw recording status
  drawText(`🎤 ${isRecording ? "Recording..." : "Mic inactive"}`, 10, canvas.height - 20, isRecording ? "lime" : "red");

  // Game over overlay
  if (gameOver) {
    ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawText("Game Over! Say 'go' to restart", canvas.width / 2 - 160, canvas.height / 2);
  }
}

// === Main loop ===
function loop() {
  update();
  render();
  requestAnimationFrame(loop);
}

// === Handle command from server (based on CNN model) ===
socket.on("command", function(command) {
  console.log("🎮 Received command:", command);
  logCommand(command);
  
  if (command === "up") {
    player.vy = jump;
    moving = true;
  } else if (command === "down") {
    player.vy += fall;
    moving = true;
  } else if (command === "stop") {
    moving = false;
  } else if (command === "go") {
    if (gameOver) resetGame();
    moving = true;
  }
});

// === Audio recording functions ===
async function initAudio() {
  try {
    // Request audio permission with specific constraints for better performance
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true, 
        autoGainControl: true,
        sampleRate: 44100  // Match your Flask app's DEVICE_RATE
      }
    });
    updateStatus("🎤 Microphone active");
    
    // Display available audio devices
    const devices = await navigator.mediaDevices.enumerateDevices();
    const audioInputs = devices.filter(device => device.kind === 'audioinput');
    console.log("📱 Available audio input devices:", audioInputs);
    
    if (audioInputs.length > 0) {
      console.log(`🎙️ Using: ${audioInputs[0].label}`);
    }
    
    // Create AudioContext
    audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: 44100  // Match your Flask app's DEVICE_RATE
    });
    
    // Setup MediaRecorder with lower latency options
    mediaRecorder = new MediaRecorder(stream, {
      mimeType: 'audio/webm;codecs=opus',  // More efficient codec
      audioBitsPerSecond: 16000            // Reduced quality for speed
    });
    
    mediaRecorder.ondataavailable = event => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };
    
    mediaRecorder.onstop = () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      audioChunks = [];
      
      // Convert to base64 and send to server
      const reader = new FileReader();
      reader.onloadend = () => {
        socket.emit("audio", reader.result);
        // Immediately start another recording cycle
        startRecording();
      };
      reader.readAsDataURL(audioBlob);
    };
    
    // Start initial recording
    startRecording();
    
  } catch (err) {
    console.error("🔴 Error accessing microphone:", err);
    updateStatus("❌ Microphone access denied");
  }
}

// Start a single recording cycle
function startRecording() {
  if (mediaRecorder && mediaRecorder.state === 'inactive') {
    isRecording = true;
    audioChunks = [];
    mediaRecorder.start();
    
    // Record for only 500ms (shorter duration)
    setTimeout(() => {
      if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        isRecording = false;
      }
    }, 500);  // Half second recording for faster response
  }
}

// Clear the old interval-based method
if (recordingInterval) {
  clearInterval(recordingInterval);
  recordingInterval = null;
}

// === Initialize game ===
window.onload = function() {
  initializeUI();
  updateStatus("🎮 Game loaded, initializing audio...");
  initAudio();
  loop();
};