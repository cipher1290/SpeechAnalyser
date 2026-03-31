# ===============================
# SESSION SETTINGS
# ===============================

# Maximum live recording time (seconds)
maxSesionDuration = 180   # 3 minutes

# Allow manual early exit
allowEarlyExit = True


# ===============================
# AUDIO SETTINGS
# ===============================

# Sampling rate (standard for speech models)
sampleRate = 16000

# Noise calibration duration (seconds)
calibrationDuration = 5


# ===============================
# CHUNK SETTINGS
# ===============================

chunkSize = 10          # seconds
chunkOverlap = 2       # seconds
chunkStep = chunkSize - chunkOverlap


# ===============================
# EMOTION SETTINGS
# ===============================

emotionLabels = [
    "happy",
    "sad",
    "angry",
    "fear",
    "stress",
    "surprise",
    "neutral"
]


# ===============================
# MODEL SETTINGS
# ===============================

# Whisper model size
whisperModelSize = "base"

# NLP summarization model
summarizationModel = "facebook/bart-large-cnn"


# ===============================
# FILE PATH SETTINGS
# ===============================

# Default folder for sample audio
sampleAudioFolder = "data/sample_audio/"

# Output directories
logFolder = "output/logs/"
reportFolder = "output/reports/"

ASSEMBLYAI_API_KEY = "28229c0c53f344fdb553037273c6205b"