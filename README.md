# Voice Command Flappy Game

![Lint-free](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg)

A containerized application that uses voice commands to control a Flappy Bird style game through machine learning.

## Team Members
-[team Memebr 4]
- [Team Member 1](https://github.com/username1)
- [Team Member 2](https://github.com/username2)
- [Team Member 3](https://github.com/username3)

## Project Description

This project implements a voice-controlled Flappy Bird game using machine learning for voice command recognition. The system consists of three main containerized components:

1. **Machine Learning Client**: Processes audio input to recognize voice commands using a CNN model trained on log-mel spectrogram features.
2. **Web App**: Provides a browser-based interface for playing the game and viewing analytics.
3. **MongoDB Database**: Stores game scores and voice command history.

Players can control the game by speaking commands like "up", "down", "stop", and "go". The machine learning model analyzes the audio input in real-time to detect these commands and control the game accordingly.

## Architecture

The system architecture consists of:

- **MongoDB Container**: Database for storing game scores and voice command history
- **Machine Learning Client Container**: Processes voice commands and runs the CNN model
- **Web App Container**: Flask application serving the game interface and dashboard

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
   docker-compose up -d
   ```

4. Access the application:
   - Web interface: http://localhost:5001
   - Dashboard: http://localhost:5001/dashboard
   - High scores: http://localhost:5001/scores

### Voice Commands

The following voice commands can be used to control the game:
- "up": Move the bird upward
- "down": Move the bird downward
- "stop": Pause the game
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
│   └── Dockerfile          # Container configuration
└── web-app/
    ├── app.py              # Flask web application
    ├── static/             # Static assets (CSS, JS)
    ├── templates/          # HTML templates
    ├── requirements.txt    # Python dependencies
    └── Dockerfile          # Container configuration
```

### Running Tests

```bash
# Test the machine learning client
cd machine-learning-client
pytest --cov=.

# Test the web app
cd ../web-app
pytest --cov=.
```

### Training the Model

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

2. Run the training script:
   ```
   docker exec -it flappy-ml-client python train_model.py
   ```

## Technical Details

### Machine Learning Model

- **Architecture**: Convolutional Neural Network (CNN)
- **Input**: Log-mel spectrogram features (128 mel bins, 44 time frames)
- **Training Data**: Voice commands ("up", "down", "left", "right", "go", "stop", "background")
- **Library**: TensorFlow 2.16.1

### Web Application

- **Framework**: Flask 2.0.3
- **Real-time Communication**: Flask-SocketIO
- **Frontend**: HTML5, CSS3, JavaScript

### Database

- **Database**: MongoDB
- **Collections**:
  - `game_scores`: Player scores with timestamps
  - `commands`: Detected voice commands

## Continuous Integration

This project uses GitHub Actions for continuous integration:

- **Lint and Format**: Checks code quality using pylint and black
- **Test and Coverage**: Runs unit tests with pytest and measures code coverage

## Development Workflow

We followed an agile development workflow:

1. **Task Tracking**: Used GitHub Projects for task organization
2. **Feature Branches**: Developed features in separate branches
3. **Pull Requests**: Code reviews conducted through PRs
4. **Regular Standups**: Daily communication to track progress and address blockers

## License

[MIT License](LICENSE)
