import os
import numpy as np
import soundfile as sf
import librosa
import config


def loadAudioFile(filePath):
    """
    Loads an audio file and converts it to standard format:
    - mono channel
    - target sample rate (from config)
    - numpy float array
    """

    # -----------------------------
    # Validate file path
    # -----------------------------
    cleanedPath = cleanFilePath(filePath)

    if not os.path.exists(cleanedPath):
        raise FileNotFoundError(f"Audio file not found: {cleanedPath}")

    # -----------------------------
    # Load audio file
    # -----------------------------
    audioData, sampleRate = sf.read(cleanedPath)

    # -----------------------------
    # Convert to mono if stereo
    # -----------------------------
    if len(audioData.shape) > 1:
        audioData = np.mean(audioData, axis=1)

    # -----------------------------
    # Resample if needed
    # -----------------------------
    if sampleRate != config.sampleRate:
        audioData = librosa.resample(
            audioData,
            orig_sr=sampleRate,
            target_sr=config.sampleRate
        )
        sampleRate = config.sampleRate

    # -----------------------------
    # Ensure float32 format
    # -----------------------------
    audioData = audioData.astype(np.float32)

    return audioData, sampleRate


def cleanFilePath(filePath):
    """
    Removes quotes and trims spaces from user input path.
    Handles Windows paste with quotes.
    """
    filePath = filePath.strip()

    if filePath.startswith('"') and filePath.endswith('"'):
        filePath = filePath[1:-1]

    return filePath