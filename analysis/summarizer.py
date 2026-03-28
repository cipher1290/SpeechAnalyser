import requests


class LLMSummarizer:

    def __init__(self):
        self.url = "http://localhost:11434/api/generate"
        self.model = "llama3:8b"

    def generateSummary(self, text, emotions):

        emotionText = ", ".join([e[0] for e in emotions[:2]])

        prompt = f"""
You are an expert speech analyst.

Analyze the following speech and generate a natural, human-like summary.

Guidelines:
- Capture emotional tone and intent
- Explain what the speaker feels and why
- Keep it meaningful and natural
- Do NOT add names or external facts
- Do NOT hallucinate anything
- Do NOT give advice
- Do NOT act as a therapist
- Do NOT respond emotionally
- Do NOT add any extra explanation

ONLY OUTPUT:
A clear, objective summary of the speech.

Focus on:
- what the speaker is saying
- emotional tone

Emotion context: {emotionText}

Speech:
{text}

Summary:
"""

        response = requests.post(self.url, json={
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7
            }
        })

        return response.json()["response"].strip()