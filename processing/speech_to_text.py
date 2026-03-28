import whisper
import torch
import config


class SpeechToTextEngine:

    def __init__(self):
        self.device = "cpu"  # Force CPU mode
        self.model = self.loadModel()

    def loadModel(self):
        print("Loading Whisper model...")
        model = whisper.load_model(config.whisperModelSize, device=self.device)
        print("Whisper model loaded successfully")
        return model

    def transcribeChunk(self, audioData, sampleRate):
        """
        Accepts numpy float32 audio array
        Returns transcription string
        """

        # Whisper expects 16kHz float32 numpy array
        result = self.model.transcribe(
            audioData,
            fp16=False  # CPU requires fp16 disabled
        )

        transcriptText = result["text"].strip()

        return transcriptText