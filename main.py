from audio.file_loader import loadAudioFile
from audio.chunker import generateAudioChunks
from processing.audio_preprocess import preprocessAudio
from processing.speech_to_text import SpeechToTextEngine
from analysis.emotion_detector import TextEmotionDetector
from analysis.trend_analyzer import EmotionTrendAnalyzer


def main():

    filePath = input("Enter audio file path: ")

    print("\nLoading audio...")
    audioData, sampleRate = loadAudioFile(filePath)

    print("Generating chunks...")
    chunks = generateAudioChunks(audioData, sampleRate)

    sttEngine = SpeechToTextEngine()
    emotionDetector = TextEmotionDetector()
    trendAnalyzer = EmotionTrendAnalyzer()

    print("\nProcessing chunks...\n")

    for chunk in chunks:

        print(f"=== Chunk {chunk.chunkId} "
              f"({chunk.startTime:.1f}s - {chunk.endTime:.1f}s) ===")

        processedAudio = preprocessAudio(chunk)

        transcript = sttEngine.transcribeChunk(
            processedAudio,
            sampleRate
        )

        print("Transcript:")
        print(transcript if transcript else "[No speech detected]")

        if transcript.strip():
            emotions = emotionDetector.detectEmotion(transcript)

            print("\nTop Emotions:")
            for label, score in emotions:
                print(f"{label} → {score:.2f}%")

            # 🔥 IMPORTANT: add to trend analyzer
            trendAnalyzer.addChunkEmotions(emotions)

        print("\n-----------------------------\n")

    # 🔥 FINAL SUMMARY OUTPUT
    print("\n===== OVERALL EMOTION ANALYSIS =====\n")

    summary = trendAnalyzer.getTrendSummary()

    # sort by percentage
    sortedSummary = sorted(summary.items(), key=lambda x: x[1], reverse=True)

    # apply threshold + top 4
    topEmotions = [e for e in sortedSummary if e[1] >= 5][:4]

    for emotion, percentage in topEmotions:
        print(f"{emotion} → {percentage:.2f}%")

    print("\nDominant Emotion:", trendAnalyzer.getDominantEmotion())


if __name__ == "__main__":
    main()