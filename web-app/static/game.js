// === Voice Flappy Game (Enhanced Modern UI with Visual Feedback) ===
const canvas = document.querySelector("canvas");
const ctx = canvas.getContext("2d");
canvas.width = 800;
canvas.height = 400;

// Game state
let player = { 
    x: 50, 
    y: 150, 
    width: 30, 
    height: 30, 
    vy: 0,
    rotation: 0,
    color: "#3b82f6" // Primary blue
};
let gravity = 0.003; // Reduced from 0.005 for slower falling
let jump = -0.4;    // Reduced from -0.5 for gentler jumps
let fall = 0.3;     // Reduced from 0.5 for gentler descents
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
let lastCommand = "";
let commandFeedback = { command: "", alpha: 0, timer: 0 };
let particles = [];
let bgParticles = [];


// Game settings
const pipeGap = 180;  
const pipeWidth = 50;
let pipeInterval = 140; // Increased from 120 to create more space between pipes
let pipeSpeed = 2;     // Reduced from 3 for slower pipe movement
let frameCount = 0;
let bgGradient;

// Connect to socket.io
const socket = io();
let lastCommandTime = 0;
const commandCooldown = 300; // milliseconds between commands to prevent duplicates
let commandBuffer = []; // buffer for recent commands to improve reliability
const bufferSize = 3; // number of commands to keep in buffer
const commandDebounceTime = 800; // milliseconds to wait before executing another of the same command

// Game colors
const colors = {
    background: "#f0f4f8",
    backgroundGradient: ["#dbeafe", "#f0f4f8"],
    player: "#3b82f6", 
    playerStroke: "#2563eb",
    pipes: "#10b981",
    pipesStroke: "#059669",
    text: "#1f2937",
    scoreText: "#3b82f6",
    commandUp: "#16a34a",
    commandDown: "#ea580c",
    commandStop: "#dc2626",
    commandGo: "#2563eb",
    particles: ["#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe"]
};

// Create a gradient background once
function createGradient() {
    bgGradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
    bgGradient.addColorStop(0, colors.backgroundGradient[0]);
    bgGradient.addColorStop(1, colors.backgroundGradient[1]);
}

// Initialize UI elements
function initializeUI() {
    createGradient();
    
    // Create background particles for parallax effect
    for (let i = 0; i < 20; i++) {
        bgParticles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            size: Math.random() * 3 + 1,
            speed: Math.random() * 0.2 + 0.1,
            alpha: Math.random() * 0.3 + 0.1
        });
    }
    
    // Status element for recording status
    if (!statusElement) {
        statusElement = document.createElement("div");
        statusElement.className = "status-bubble";
        document.body.appendChild(statusElement);
    }
}

// Update status message
function updateStatus(text) {
    if (statusElement) {
        statusElement.textContent = text;
        statusElement.className = "status-bubble fade-in";
        // Reset animation
        void statusElement.offsetWidth;
        statusElement.className = "status-bubble fade-in";
    }
}

// Draw text with shadow
function drawText(text, x, y, color = "#1f2937", size = 22, align = "left") {
    ctx.save();
    ctx.fillStyle = color;
    ctx.font = `${size}px 'Segoe UI', sans-serif`;
    ctx.textAlign = align;
    
    // Draw shadow
    ctx.shadowColor = "rgba(0,0,0,0.1)";
    ctx.shadowBlur = 4;
    ctx.shadowOffsetX = 1;
    ctx.shadowOffsetY = 1;
    
    ctx.fillText(text, x, y);
    ctx.restore();
}

// Create a new pipe
function createPipe() {
    let topHeight = Math.floor(Math.random() * (canvas.height - pipeGap - 60)) + 20;
    pipes.push({ 
        x: canvas.width, 
        top: topHeight, 
        passed: false,
        color: colors.pipes,
        strokeColor: colors.pipesStroke 
    });
}

// Reset game state
function resetGame() {
    player.y = 150;
    player.vy = 0;
    player.rotation = 0;
    pipes = [];
    score = 0;
    moving = true;
    gameOver = false;
    particles = [];
}

// Check collision between player and pipe
function checkCollision(pipe) {
    const inPipe = player.x + player.width > pipe.x && player.x < pipe.x + pipeWidth;
    const hitPipe = player.y < pipe.top || player.y + player.height > pipe.top + pipeGap;
    return inPipe && hitPipe;
}

// Send score to server
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

// Create particles effect
function createParticles(x, y, count, color) {
    for (let i = 0; i < count; i++) {
        particles.push({
            x: x,
            y: y,
            vx: (Math.random() - 0.5) * 2,
            vy: (Math.random() - 0.5) * 2,
            size: Math.random() * 3 + 2,
            color: color || colors.particles[Math.floor(Math.random() * colors.particles.length)],
            alpha: 1,
            life: Math.random() * 30 + 20
        });
    }
}

// Update game state
function update() {
    // Update player
    if (moving) {
        player.vy += gravity;
        player.y += player.vy;
        
        // Update player rotation based on velocity
        player.rotation = player.vy * 20;
        if (player.rotation > Math.PI/6) player.rotation = Math.PI/6;
        if (player.rotation < -Math.PI/6) player.rotation = -Math.PI/6;
        
        // Bound checking
        if (player.y < 0) {
            player.y = 0;
            player.vy = 0;
        }
    }
    
    // Create new pipes at intervals
    if (frameCount % pipeInterval === 0) createPipe();
    
    // Update pipes
    pipes.forEach(pipe => pipe.x -= pipeSpeed); // Use the pipeSpeed variable instead of fixed value
    pipes = pipes.filter(pipe => pipe.x + pipeWidth > 0);
    
    // Check for score and collisions
    pipes.forEach(pipe => {
        if (pipe.x + pipeWidth < player.x && !pipe.passed) {
            pipe.passed = true;
            score++;
            // Create score particles effect
            createParticles(player.x + player.width, player.y, 10);
        }
        
        if (!gameOver && checkCollision(pipe)) {
            gameOver = true;
            moving = false;
            sendScoreToServer(score);
            // Create crash particles
            createParticles(player.x, player.y, 20, "#ef4444");
        }
    });
    
    // Check if player hit the ground
    if (!gameOver && player.y > canvas.height - player.height) {
        player.y = canvas.height - player.height;
        gameOver = true;
        moving = false;
        sendScoreToServer(score);
        // Create crash particles
        createParticles(player.x, player.y, 20, "#ef4444");
    }
    
    // Update command feedback
    if (commandFeedback.alpha > 0) {
        commandFeedback.alpha -= 0.02;
    }
    if (commandFeedback.timer > 0) {
        commandFeedback.timer--;
    } else {
        commandFeedback.command = "";
    }
    
    // Update particles
    for (let i = particles.length - 1; i >= 0; i--) {
        let p = particles[i];
        p.x += p.vx;
        p.y += p.vy;
        p.life--;
        p.alpha = p.life / 50;
        
        if (p.life <= 0) {
            particles.splice(i, 1);
        }
    }
    
    // Update background particles
    bgParticles.forEach(p => {
        p.x -= p.speed;
        if (p.x < 0) {
            p.x = canvas.width;
            p.y = Math.random() * canvas.height;
        }
    });
    
    frameCount++;
}

// Render game graphics
function render() {
    // Draw background
    ctx.fillStyle = bgGradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw background particles for depth effect
    bgParticles.forEach(p => {
        ctx.fillStyle = `rgba(255, 255, 255, ${p.alpha})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();
    });
    
    // Draw pipes
    ctx.strokeStyle = colors.pipesStroke;
    ctx.lineWidth = 2;
    
    pipes.forEach(pipe => {
        // Top pipe
        ctx.fillStyle = colors.pipes;
        ctx.beginPath();
        ctx.roundRect(pipe.x, 0, pipeWidth, pipe.top, [0, 0, 6, 6]);
        ctx.fill();
        ctx.stroke();
        
        // Bottom pipe
        ctx.beginPath();
        ctx.roundRect(pipe.x, pipe.top + pipeGap, pipeWidth, canvas.height - pipe.top - pipeGap, [6, 6, 0, 0]);
        ctx.fill();
        ctx.stroke();
    });
    
    // Draw player with rotation
    ctx.save();
    ctx.translate(player.x + player.width/2, player.y + player.height/2);
    ctx.rotate(player.rotation);
    
    // Player shadow
    ctx.fillStyle = "rgba(0,0,0,0.1)";
    ctx.beginPath();
    ctx.roundRect(-player.width/2 + 2, -player.height/2 + 2, player.width, player.height, 8);
    ctx.fill();
    
    // Player body
    ctx.fillStyle = colors.player;
    ctx.strokeStyle = colors.playerStroke;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(-player.width/2, -player.height/2, player.width, player.height, 8);
    ctx.fill();
    ctx.stroke();
    
    // Player details (eye and wing)
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.arc(-player.width/6, -player.height/6, player.width/8, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.fillStyle = "#000";
    ctx.beginPath();
    ctx.arc(-player.width/6, -player.height/6, player.width/16, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.restore();
    
    // Draw particles
    particles.forEach(p => {
        ctx.fillStyle = `rgba(${hexToRgb(p.color)}, ${p.alpha})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();
    });
    
    // Draw score
    drawText(`Score: ${score}`, 20, 40, colors.scoreText, 28);
    
    // Draw recording status with pulsing effect for "Listening"
    if (isRecording) {
        ctx.fillStyle = "rgba(22, 163, 74, 0.2)"; // Light green background
        ctx.beginPath();
        ctx.roundRect(10, canvas.height - 40, 130, 30, 15);
        ctx.fill();
        
        // Pulsing dots for recording indication
        const time = Date.now() / 500;
        for (let i = 0; i < 3; i++) {
            const alpha = 0.4 + 0.6 * Math.sin((time + i * 0.5) % Math.PI);
            ctx.fillStyle = `rgba(22, 163, 74, ${alpha})`;
            ctx.beginPath();
            ctx.arc(100 + i * 8, canvas.height - 25, 3, 0, Math.PI * 2);
            ctx.fill();
        }
        
        drawText(`🎤 Listening`, 20, canvas.height - 20, colors.commandGo, 16);
    } else {
        drawText(`🎤 Mic off`, 20, canvas.height - 20, "#888", 16);
    }
    
    // Draw command feedback
    if (commandFeedback.command && commandFeedback.alpha > 0) {
        let cmdColor;
        switch (commandFeedback.command) {
            case "up": cmdColor = colors.commandUp; break;
            case "down": cmdColor = colors.commandDown; break;
            case "stop": cmdColor = colors.commandStop; break;
            case "go": cmdColor = colors.commandGo; break;
            default: cmdColor = colors.text;
        }
        
        ctx.fillStyle = `rgba(255, 255, 255, ${commandFeedback.alpha * 0.7})`;
        ctx.beginPath();
        ctx.roundRect(canvas.width/2 - 60, 20, 120, 40, 20);
        ctx.fill();
        
        drawText(commandFeedback.command.toUpperCase(), canvas.width/2, 45, cmdColor, 24, "center");
    }
    
    // Draw game over overlay
    if (gameOver) {
        ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw game over card
        ctx.fillStyle = "#fff";
        ctx.beginPath();
        ctx.roundRect(canvas.width/2 - 150, canvas.height/2 - 80, 300, 160, 12);
        ctx.fill();
        
        drawText("Game Over", canvas.width/2, canvas.height/2 - 30, colors.text, 32, "center");
        drawText(`Score: ${score}`, canvas.width/2, canvas.height/2 + 10, colors.scoreText, 28, "center");
        drawText("Say 'go' to restart", canvas.width/2, canvas.height/2 + 50, "#666", 20, "center");
    }
}

// Game loop variables for fixed time step
let lastTime = 0;
const fixedDeltaTime = 1000/60; // Target: 60 FPS
let accumulator = 0;

// Game loop with fixed time step
function loop(timestamp) {
    if (!lastTime) lastTime = timestamp;
    
    // Calculate delta time between frames in ms
    const deltaTime = timestamp - lastTime;
    lastTime = timestamp;
    
    // Add to the accumulator
    accumulator += deltaTime;
    
    // Update with a fixed time step to ensure consistent speed across devices
    while (accumulator >= fixedDeltaTime) {
        update();
        accumulator -= fixedDeltaTime;
    }
    
    // Render at whatever frame rate the browser provides
    render();
    
    // Continue the loop
    requestAnimationFrame(loop);
}

// Process voice commands
socket.on("command", function(command) {
    console.log("🎮 Received command:", command);
    
    const now = Date.now();
    
    // Ignore rapid repeats of the same command
    if (now - lastCommandTime < commandCooldown) {
        console.log("Command ignored due to cooldown");
        return;
    }
    
    // Only process if it's a recognized command
    if (["up", "down", "stop", "go"].includes(command)) {
        // Add command to buffer
        commandBuffer.push({ command, time: now });
        if (commandBuffer.length > bufferSize) {
            commandBuffer.shift(); // remove oldest command
        }
        
        // Check for consistent commands in buffer
        const recentCommands = {};
        commandBuffer.forEach(c => {
            if (!recentCommands[c.command]) recentCommands[c.command] = 0;
            // Only count commands within the last second
            if (now - c.time < 1000) recentCommands[c.command]++;
        });
        
        // Display command feedback
        commandFeedback.command = command;
        commandFeedback.alpha = 1;
        commandFeedback.timer = 40;
        
        // Update last command time
        lastCommandTime = now;
        
        // Handle the command with improved logic
        if (command === "up") {
            // Only apply a new jump if we're not already moving up rapidly
            if (player.vy > -0.2) {
                player.vy = jump;
            }
            moving = true;
            createParticles(player.x, player.y + player.height, 5, "#60a5fa");
        } else if (command === "down") {
            player.vy += fall;
            moving = true;
            createParticles(player.x, player.y - player.height/2, 5, "#f97316");
        } else if (command === "stop") {
            moving = false;
        } else if (command === "go") {
            if (gameOver) resetGame();
            moving = true;
            createParticles(player.x, player.y, 10, "#10b981");
        }
    }
});

// Initialize audio recording
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
                setTimeout(startRecording, 500); // Reduced delay between recordings from 800ms to 500ms
            };
            reader.readAsDataURL(audioBlob);
        };

        startRecording();
    } catch (err) {
        console.error("🔴 Error accessing microphone:", err);
        updateStatus("❌ Microphone access denied");
    }
}

// Start recording audio with longer duration
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
        }, 800); // Increased from 800ms to 1500ms (1.5 seconds)
    }
}

// Utility function to convert hex to rgb
function hexToRgb(hex) {
    // Remove the # if present
    hex = hex.replace("#", "");
    
    // Parse the hex values
    const r = parseInt(hex.substring(0, 2), 16);
    const g = parseInt(hex.substring(2, 4), 16);
    const b = parseInt(hex.substring(4, 6), 16);
    
    return `${r}, ${g}, ${b}`;
}

// Initialize game
window.onload = function() {
    initializeUI();
    updateStatus("🎮 Game loaded, initializing audio...");
    initAudio();
    
    // Start the game loop with timestamp
    requestAnimationFrame(loop);
    
    // Show welcome message
    setTimeout(() => {
        updateStatus("Say 'up', 'down', 'stop', or 'go' to play!");
    }, 3000);
};
