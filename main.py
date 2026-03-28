from audio.file_loader import loadAudioFile
from audio.chunker import generateAudioChunks
from processing.audio_preprocess import preprocessAudio
from processing.speech_to_text import SpeechToTextEngine
from analysis.emotion_detector import TextEmotionDetector
from analysis.trend_analyzer import EmotionTrendAnalyzer
from processing.transcript_merger import mergeTranscripts
from analysis.summarizer import LLMSummarizer

def main():

    filePath = input("Enter audio file path: ")

    print("\nLoading audio...")
    audioData, sampleRate = loadAudioFile(filePath)

    print("Generating chunks...")
    chunks = generateAudioChunks(audioData, sampleRate)

    sttEngine = SpeechToTextEngine()
    emotionDetector = TextEmotionDetector()
    trendAnalyzer = EmotionTrendAnalyzer()
    llmSummarizer = LLMSummarizer()

    print("\nProcessing chunks...\n")

    allTranscripts = []   # 🔥 ONLY THIS needed

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

        # 🔥 ADD THIS (IMPORTANT)
        if transcript.strip():
            allTranscripts.append(transcript)

            emotions = emotionDetector.detectEmotion(transcript)

            print("\nTop Emotions:")
            for label, score in emotions:
                print(f"{label} → {score:.2f}%")

            trendAnalyzer.addChunkEmotions(emotions)

        print("\n-----------------------------\n")

    # 🔥 OVERALL EMOTION ANALYSIS
    print("\n===== OVERALL EMOTION ANALYSIS =====\n")

    summary = trendAnalyzer.getTrendSummary()

    sortedSummary = sorted(summary.items(), key=lambda x: x[1], reverse=True)

    topEmotions = [e for e in sortedSummary if e[1] >= 5][:4]

    for emotion, percentage in topEmotions:
        print(f"{emotion} → {percentage:.2f}%")

    print("\nDominant Emotion:", trendAnalyzer.getDominantEmotion())

    # 🔥 FINAL MERGED TRANSCRIPT
    finalText = mergeTranscripts(allTranscripts)

    print("\n===== FINAL TRANSCRIPT =====\n")
    print(finalText)

    print("\n===== SUMMARY =====\n")

    summaryText = llmSummarizer.generateSummary(finalText, topEmotions)

    print(summaryText)

if __name__ == "__main__":
    main()