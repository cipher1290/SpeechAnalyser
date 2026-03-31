from audio.file_loader import loadAudioFile
# from audio.chunker import generateAudioChunks
from audio.chunker import splitText
from processing.audio_preprocess import preprocessAudio
# from processing.speech_to_text import SpeechToTextEngine
from analysis.trend_analyzer import EmotionTrendAnalyzer
# from processing.transcript_merger import mergeTranscripts
# from processing.transcript_cleaner import TranscriptCleaner
from analysis.summarizer import LLMSummarizer
from analysis.emotion_detector import RobertaEmotionDetector
from processing.assemblyai_transcript import AssemblySTT

def main():

    filePath = input("Enter audio file path: ")

    print("\nLoading audio...")
    audioData, sampleRate = loadAudioFile(filePath)

    # print("Generating chunks...")
    # chunks = generateAudioChunks(audioData, sampleRate)

    # sttEngine = SpeechToTextEngine()
    emotionDetector =RobertaEmotionDetector()
    trendAnalyzer = EmotionTrendAnalyzer()
    llmSummarizer = LLMSummarizer()
    # cleaner = TranscriptCleaner()
    sttEngine = AssemblySTT()

    print("\nProcessing chunks...\n")

    fullTranscript = sttEngine.transcribe(filePath)

    textChunks = splitText(fullTranscript)
    chunkCount = 1
    for chunk in textChunks:
        print(f"=== Chunk {chunkCount} ===")
        print(chunk if chunk else "[No speech detected]")
        emotions = emotionDetector.detectEmotion(chunk)
        print("\nTop Emotions:")
        for label, score in emotions:
            print(f"{label} → {score:.2f}%")

        trendAnalyzer.addChunkEmotions(emotions)

        print("\n-----------------------------\n")
        chunkCount = chunkCount + 1


    # allTranscripts = []   # 🔥 ONLY THIS needed

    # for chunk in chunks:

    #     # print(f"=== Chunk {chunk.chunkId} "
    #     #       f"({chunk.startTime:.1f}s - {chunk.endTime:.1f}s) ===")

    #     processedAudio = preprocessAudio(chunk)

    #     transcript = sttEngine.transcribeChunk(
    #         processedAudio,
    #         sampleRate
    #     )

    #     print("Transcript:")
    #     print(transcript if transcript else "[No speech detected]")

    #     # 🔥 ADD THIS (IMPORTANT)
    #     if transcript.strip():
    #         allTranscripts.append(transcript)

    #         emotions = emotionDetector.detectEmotion(transcript)

    #         print("\nTop Emotions:")
    #         for label, score in emotions:
    #             print(f"{label} → {score:.2f}%")

    #         trendAnalyzer.addChunkEmotions(emotions)

    #     print("\n-----------------------------\n")

    # OVERALL EMOTION ANALYSIS
    print("\n===== OVERALL EMOTION ANALYSIS =====\n")

    summary = trendAnalyzer.getTrendSummary()

    sortedSummary = sorted(summary.items(), key=lambda x: x[1], reverse=True)

    topEmotions = [e for e in sortedSummary if e[1] >= 5][:4]

    for emotion, percentage in topEmotions:
        print(f"{emotion} → {percentage:.2f}%")

    print("\nDominant Emotion:", trendAnalyzer.getDominantEmotion())

    # FINAL MERGED TRANSCRIPT
    # rawText = mergeTranscripts(allTranscripts)
    
    # print("\n===== INITIAL TRANSCRIPT =====\n")

    # print(rawText)
    # print("\nCleaning transcript with LLM...\n")

    # finalText = cleaner.clean(rawText)

    print("\n===== FINAL TRANSCRIPT =====\n")
    print(fullTranscript)

    print("\n===== SUMMARY =====\n")

    summaryText = llmSummarizer.generateSummary(fullTranscript, topEmotions)

    print(summaryText)

if __name__ == "__main__":
    main()