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

        # sort already assumed, but ensure
        emotions = sorted(emotions, key=lambda x: x[1], reverse=True)

        topLabel = emotions[0][0]
        topScore = emotions[0][1]

        # 🔥 FIX 1: neutral suppression
        if topLabel == "neutral" and len(emotions) > 1:
            secondLabel, secondScore = emotions[1]

            # if neutral not strongly dominant → ignore it
            if (topScore - secondScore) < 10:
                topLabel = secondLabel
                topScore = secondScore

        # 🔥 FIX 2: weighted aggregation (ignore weak neutral)
        for label, score in emotions:

            if label == "neutral" and score < 70:
                continue

            if score >= self.threshold:
                self.emotionScores[label] += score

        # 🔥 FIX 3: clean flow tracking
        self.chunkFlow.append(topLabel if topLabel else "neutral")

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

        # 🔥 FIX 4: avoid neutral dominance
        sortedEmotions = sorted(summary.items(), key=lambda x: x[1], reverse=True)

        for label, _ in sortedEmotions:
            if label != "neutral":
                return label

        return sortedEmotions[0][0]

    def getEmotionFlow(self):
        return self.chunkFlow