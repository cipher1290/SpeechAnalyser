from collections import defaultdict


class EmotionTrendAnalyzer:

    def __init__(self, threshold=5.0):
        self.threshold = threshold
        self.emotionScores = defaultdict(float)
        self.chunkFlow = []

    def addChunkEmotions(self, emotions):
        """
        emotions = list of tuples:
        [("anger", 68.8), ("surprise", 26.3), ("sadness", 3.4)]
        """
        topLabel = None
        topScore = 0
        # pick top emotion above threshold
        for label, score in emotions:

            # track top emotion for flow
            if score > topScore:
                topScore = score
                topLabel = label

            # apply threshold to avoid noise
            if score >= self.threshold:
                self.emotionScores[label] += score

        # flow tracking (for timeline)
        if topLabel:
            self.chunkFlow.append(topLabel)
        else:
            self.chunkFlow.append("neutral")

    def getTrendSummary(self):

        if not self.emotionScores:
            return {}

        totalScore = sum(self.emotionScores.values())

        summary = {}

        for emotion, score in self.emotionScores.items():
            percentage = (score / totalScore) * 100
            summary[emotion] = percentage

        return summary

    def getDominantEmotion(self):
        summary = self.getTrendSummary()

        if not summary:
            return "neutral"

        return max(summary, key=summary.get)