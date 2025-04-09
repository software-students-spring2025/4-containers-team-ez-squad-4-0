// === Voice Flappy Game (Minimal, Modern Soft UI with Score Save) ===
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

const pipeGap = 180;
const pipeWidth = 50;
let pipeInterval = 100;
let frameCount = 0;

const socket = io();

function initializeUI() {
  if (!statusElement) {
    statusElement = document.createElement("div");
    statusElement.style.position = "absolute";
    statusElement.style.bottom = "20px";
    statusElement.style.left = "50%";
    statusElement.style.transform = "translateX(-50%)";
    statusElement.style.color = "#444";
    statusElement.style.background = "rgba(255,255,255,0.7)";
    statusElement.style.padding = "10px 18px";
    statusElement.style.borderRadius = "16px";
    statusElement.style.fontWeight = "600";
    statusElement.style.fontFamily = "'Segoe UI', sans-serif";
    statusElement.style.boxShadow = "0 4px 12px rgba(0,0,0,0.15)";
    document.body.appendChild(statusElement);
  }
}

function updateStatus(text) {
  if (statusElement) statusElement.textContent = text;
}

function drawText(text, x, y, color = "#333", size = 22) {
  ctx.fillStyle = color;
  ctx.font = `${size}px 'Segoe UI', sans-serif`;
  ctx.fillText(text, x, y);
}

function createPipe() {
  let topHeight = Math.floor(Math.random() * (canvas.height - pipeGap - 60)) + 20;
  pipes.push({ x: canvas.width, top: topHeight, passed: false });
}

function resetGame() {
  player.y = 150;
  player.vy = 0;
  pipes = [];
  score = 0;
  moving = true;
  gameOver = false;
}

function checkCollision(pipe) {
  const inPipe = player.x + player.width > pipe.x && player.x < pipe.x + pipeWidth;
  const hitPipe = player.y < pipe.top || player.y + player.height > pipe.top + pipeGap;
  return inPipe && hitPipe;
}

function sendScoreToServer(score) {
  fetch("/score", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ score })
  }).then(res => {
    if (res.ok) {
      console.log("✅ Score submitted:", score);
    } else {
      console.error("❌ Failed to submit score.");
    }
  });
}

function update() {
  if (moving) {
    player.vy += gravity;
    player.y += player.vy;
    if (player.y < 0) {
      player.y = 0;
      player.vy = 0;
    }
  }
  if (frameCount % pipeInterval === 0) createPipe();
  pipes.forEach(pipe => pipe.x -= 3);
  pipes = pipes.filter(pipe => pipe.x + pipeWidth > 0);
  pipes.forEach(pipe => {
    if (pipe.x + pipeWidth < player.x && !pipe.passed) {
      pipe.passed = true;
      score++;
    }
    if (!gameOver && checkCollision(pipe)) {
      gameOver = true;
      moving = false;
      sendScoreToServer(score);
    }
  });
  if (!gameOver && player.y > canvas.height) {
    gameOver = true;
    moving = false;
    sendScoreToServer(score);
  }
  frameCount++;
}

function render() {
  ctx.fillStyle = "#f0f4f8";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "#3b82f6";
  ctx.beginPath();
  ctx.roundRect(player.x, player.y, player.width, player.height, 6);
  ctx.fill();

  ctx.fillStyle = "#10b981";
  pipes.forEach(pipe => {
    ctx.fillRect(pipe.x, 0, pipeWidth, pipe.top);
    ctx.fillRect(pipe.x, pipe.top + pipeGap, pipeWidth, canvas.height);
  });

  drawText(`Score: ${score}`, 10, 30);
  drawText(`🎤 ${isRecording ? "Listening..." : "Mic off"}`, 10, canvas.height - 20, isRecording ? "#16a34a" : "#888");

  if (gameOver) {
    ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawText("Game Over", canvas.width / 2 - 80, canvas.height / 2 - 10, "#fff", 28);
    drawText("Say 'go' to restart", canvas.width / 2 - 100, canvas.height / 2 + 24, "#fff", 20);
  }
}

function loop() {
  update();
  render();
  requestAnimationFrame(loop);
}

socket.on("command", function(command) {
  console.log("🎮 Received command:", command);

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

async function initAudio() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        sampleRate: 44100
      }
    });
    updateStatus("🎤 Microphone active");

    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 44100 });
    mediaRecorder = new MediaRecorder(stream, {
      mimeType: 'audio/webm;codecs=opus',
      audioBitsPerSecond: 16000
    });

    mediaRecorder.ondataavailable = event => {
      if (event.data.size > 0) audioChunks.push(event.data);
    };

    mediaRecorder.onstop = () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      audioChunks = [];
      const reader = new FileReader();
      reader.onloadend = () => {
        socket.emit("audio", reader.result);
        setTimeout(startRecording, 800);
      };
      reader.readAsDataURL(audioBlob);
    };

    startRecording();
  } catch (err) {
    console.error("🔴 Error accessing microphone:", err);
    updateStatus("❌ Microphone access denied");
  }
}

function startRecording() {
  if (mediaRecorder && mediaRecorder.state === 'inactive') {
    isRecording = true;
    audioChunks = [];
    mediaRecorder.start();
    setTimeout(() => {
      if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        isRecording = false;
      }
    }, 800);
  }
}

if (recordingInterval) clearInterval(recordingInterval);
recordingInterval = null;

window.onload = function () {
  initializeUI();
  updateStatus("🎮 Game loaded, initializing audio...");
  initAudio();
  loop();
};