import numpy as np
import config


class AudioChunk:
    def __init__(self, chunkId, startTime, endTime, audioData, sampleRate):
        self.chunkId = chunkId
        self.startTime = startTime
        self.endTime = endTime
        self.audioData = audioData
        self.sampleRate = sampleRate


def generateAudioChunks(audioData, sampleRate):
    """
    Splits full audio into overlapping chunks based on config settings.
    Returns list of AudioChunk objects.
    """

    chunkSizeSamples = int(config.chunkSize * sampleRate)
    chunkStepSamples = int(config.chunkStep* sampleRate)

    totalSamples = len(audioData)
    currentStart = 0
    chunkId = 1

    chunks = []

    while currentStart < totalSamples:

        currentEnd = currentStart + chunkSizeSamples

        # If last chunk shorter than chunk size → pad with zeros
        if currentEnd > totalSamples:
            chunkAudio = audioData[currentStart:totalSamples]
            paddingNeeded = currentEnd - totalSamples
            chunkAudio = np.pad(chunkAudio, (0, paddingNeeded))
        else:
            chunkAudio = audioData[currentStart:currentEnd]

        startTime = currentStart / sampleRate
        endTime = currentEnd / sampleRate

        chunkObject = AudioChunk(
            chunkId=chunkId,
            startTime=startTime,
            endTime=endTime,
            audioData=chunkAudio,
            sampleRate=sampleRate
        )

        chunks.append(chunkObject)

        currentStart += chunkStepSamples
        chunkId += 1

    return chunks