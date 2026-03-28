import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class TextEmotionDetector:

    def __init__(self):
        self.device = "cpu"
        self.tokenizer = None
        self.model = None
        self.loadModel()

    def loadModel(self):
        print("Loading GoEmotions model...")

        modelName = "monologg/bert-base-cased-goemotions-original"

        self.tokenizer = AutoTokenizer.from_pretrained(modelName)
        self.model = AutoModelForSequenceClassification.from_pretrained(modelName)

        self.model.to(self.device)
        self.model.eval()

        print("GoEmotions model loaded successfully")

    def detectEmotion(self, text):

        if not text.strip():
            return [("neutral", 100.0)]

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probabilities = torch.sigmoid(logits).squeeze()

        labels = self.model.config.id2label

        emotionScores = []

        for i in range(len(probabilities)):
            emotionScores.append(
                (labels[i], float(probabilities[i]) * 100)
            )

        # Sort by probability descending
        emotionScores.sort(key=lambda x: x[1], reverse=True)

        # 🔥 Remove weak neutral dominance
        filtered = []

        for label, score in emotionScores:
            if label == "neutral" and score < 70:
                continue
            filtered.append((label, score))

        # fallback if everything removed
        if not filtered:
            filtered = emotionScores

        return filtered[:3]