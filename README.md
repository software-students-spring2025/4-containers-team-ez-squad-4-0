![ML Client Tests](https://github.com/software-students-spring2025/4-containers-team-ez-squad-4-0/actions/workflows/ml-client-ci.yml/badge.svg)

![Web App Tests](https://github.com/software-students-spring2025/4-containers-team-ez-squad-4-0/actions/workflows/web-app-ci.yml/badge.svg)

![Lint-free](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg)

# Voice Command Flappy Game


A containerized application that uses voice commands to control a Flappy Bird style game through machine learning.

## Team Members

- [ChenJun Hsu](https://github.com/Junpapadiamond)
- [Eric Zhao](https://github.com/Ericzzy675)
- [ShenRui Xue](https://github.com/ShenruiXue666)
- [Jess Liang](https://github.com/jess-liang322)

## Project Description

This project implements a voice-controlled Flappy Bird game using machine learning for voice command recognition. The system consists of three main containerized components:

1. **Machine Learning Client**: Processes audio input to recognize voice commands using a CNN model trained on log-mel spectrogram features.
2. **Web App**: Provides a browser-based interface for playing the game and viewing analytics.
3. **MongoDB Database**: Stores game scores and voice command history.

Players can control the game by speaking commands like "up", "down", "stop", and "go". The machine learning model analyzes the audio input in real-time to detect these commands and control the game accordingly.

## Architecture

The system architecture consists of:

- **MongoDB Container**  
  Acts as the central database for storing voice command predictions and game scores.  
  - **Database**: `voice_flappy_game`  
  - **Collections**:
    - `commands`
      - `command`: Predicted voice command (e.g., "up", "stop", "background")
      - `confidence`: Model confidence score (float)
      - `timestamp`: Time when the prediction was made
      - `file_path`: File path if batch processed via `client.py`
      - `processed`: Boolean flag for processing status
    - `game_scores`
      - `score`: Final score submitted from the game
      - `timestamp`: Time the score was recorded

- **Machine Learning Client Container**  
  Processes recorded audio input and predicts the associated command using a CNN model.  
  - Loads and uses `cnn_model.h5` and `cnn_label_encoder.pkl`
  - Extracts MFCC features using `librosa`  
  - Supports batch prediction via CLI:
    - `--process-file <path>`: Predict one audio file
    - `--process-dir <dir>`: Predict all `.wav` files in a folder
  - Saves prediction results to MongoDB under `commands`

- **Web App Container**  
  A Flask + Flask-SocketIO application that hosts the game interface and live prediction dashboard.  
  - Accepts base64 audio over Socket.IO, decodes and converts it to `.wav`
  - Uses the same ML model to predict voice commands in real-time
  - Emits predictions back to the front end immediately
  - Stores commands and scores to MongoDB
  - Routes:
    - `/`: Main game page
    - `/scores`: View recent scores
    - `/score`: POST endpoint for saving score
    - `/api/commands`: Return recent command predictions in JSON
    - `/dashboard`: Web dashboard for visualization


## Setup Instructions

### Prerequisites

- Docker and Docker Compose
- Git

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/4-containers-team-ez-squad-4-0.git
   cd 4-containers-team-ez-squad-4-0
   ```

2. Environment Setup:
   Create a `.env` file in the root directory with the following variables:
   ```
   MONGO_INITDB_ROOT_USERNAME=admin
   MONGO_INITDB_ROOT_PASSWORD=password
   SECRET_KEY=your_secret_key
   ```

3. Build and start the containers:
   ```
   docker-compose build
   docker-compose up
   ```

4. Access the application:
   - Web interface: http://localhost:5001
   - Voice Analysis Dashboard: http://localhost:5001/dashboard
   - High scores: http://localhost:5001/scores

### Voice Commands

The following voice commands can be used to control the game:
- "up": Move the bird upward
- "down": Move the bird downward
- "stop": Freeze the character
- "go": Start or restart the game

## Development

### Project Structure

```
├── docker-compose.yml      # Docker Compose configuration
├── .env                    # Environment variables (create this file)
├── dataset/                # Voice command training data
├── machine-learning-client/
│   ├── Client.py           # ML client implementation
│   ├── extract_features.py # Feature extraction for audio
│   ├── train_model.py      # CNN model training
│   ├── requirements.txt    # Python dependencies
│   ├── test_client.py      # Test Client.py
│   └── Dockerfile          # Container configuration
└── web-app/
    ├── app.py              # Flask web application
    ├── static/             # Static assets (CSS, JS)
    ├── templates/          # HTML templates
    ├── requirements.txt    # Python dependencies
    ├── test_app.py         # Test app.py
    └── Dockerfile          # Container configuration
```

## Running Tests
To test both the Machine Learning client and the Web App components, follow the steps below.

### Test the Machine Learning Client

```bash
cd machine-learning-client

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests with coverage
coverage run --source=Client -m pytest test_client.py
coverage report
```

### Test the Web App


```bash
cd ../web-app/tests

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install web app test dependencies
pip install -r requirements-test.txt

# Run tests with coverage
python -m pytest tests/ -v --cov=app --cov-report=term-missing
```

### Training the Model
 **The full raw audio dataset used for training is stored on [Google Drive](https://drive.google.com/drive/folders/1Goo-JfKp5nqPBsH-AYdKKXvPrizOOndY?usp=sharing).**  
To train the machine learning model on your own dataset:

1. Place your audio samples in the `dataset` directory, organized by command:
   ```
   dataset/
   ├── up/
   │   ├── sample1.wav
   │   ├── sample2.wav
   │   └── ...
   ├── down/
   │   └── ...
   └── ...
   ```

2. **Extract features** from your dataset:
   ```
   cd machine-learning-client
   python extract_features.py
   ```
   This will process all .wav files in the dataset/ folder

3. Train the model using the extracted features:
   ```
   python train_model.py
   ```
   This output should be an h5 and a pkl file.

## babble
   1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181
182
183
184
185
186
187
188
189
190
191
192
193
194
195
196
197
198
199
200

## License

[MIT License](LICENSE)
