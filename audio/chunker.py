import numpy as np
import config


class AudioChunk:
    def __init__(self, chunkId, startTime, endTime, audioData, sampleRate):
        self.chunkId = chunkId
        self.startTime = startTime
        self.endTime = endTime
        self.audioData = audioData
        self.sampleRate = sampleRate

def splitText(text, size=2):
    sentences = text.split(".")
    chunks = []

    for i in range(0, len(sentences), size):
        chunk = ". ".join(sentences[i:i+size])
        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks