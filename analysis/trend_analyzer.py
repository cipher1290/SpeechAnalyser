from collections import Counter


class EmotionTrendAnalyzer:

    def __init__(self, threshold=5.0):
        self.threshold = threshold
        self.chunkEmotions = []

    def addChunkEmotions(self, emotions):
        """
        emotions = list of tuples:
        [("anger", 68.8), ("surprise", 26.3), ("sadness", 3.4)]
        """

        # pick top emotion above threshold
        for label, score in emotions:
            if score >= self.threshold:
                self.chunkEmotions.append(label)
                return

        # if none above threshold → neutral
        self.chunkEmotions.append("neutral")

    def getTrendSummary(self):

        if not self.chunkEmotions:
            return {}

        totalChunks = len(self.chunkEmotions)

        count = Counter(self.chunkEmotions)

        summary = {}

        for emotion, freq in count.items():
            percentage = (freq / totalChunks) * 100
            summary[emotion] = percentage

        return summary

    def getDominantEmotion(self):
        summary = self.getTrendSummary()

        if not summary:
            return "neutral"

        return max(summary, key=summary.get)

    def getEmotionFlow(self):
        return self.chunkEmotions