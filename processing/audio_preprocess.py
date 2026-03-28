import numpy as np


def normalizeAudio(audioData):
    maxAmplitude = np.max(np.abs(audioData))

    if maxAmplitude == 0:
        return audioData

    normalizedAudio = audioData / maxAmplitude
    return normalizedAudio


def preprocessAudio(audioChunk):
    audioData = audioChunk.audioData
    audioData = normalizeAudio(audioData)
    return audioData