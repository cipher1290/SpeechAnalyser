import torch
import torch.nn as nn
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from huggingface_hub import hf_hub_download
import json


class RobertaForMultiLabelClassification(nn.Module):
    def __init__(self, model_name, num_labels, dropout_rate=0.3, use_mean_pooling=True):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.use_mean_pooling = use_mean_pooling
        hidden_size = self.roberta.config.hidden_size

        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size // 2, num_labels)

    def mean_pooling(self, token_embeddings, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        return summed / counts

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)

        if self.use_mean_pooling:
            pooled = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        else:
            pooled = outputs.pooler_output

        x = self.dropout1(pooled)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)

        return self.fc2(x)


class RobertaEmotionDetector:

    def __init__(self):
        print("Loading RoBERTa Large GoEmotions model...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "Lakssssshya/roberta-large-goemotions"

        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)

        # load config
        config_path = hf_hub_download(repo_id=model_name, filename="config.json")
        with open(config_path) as f:
            config = json.load(f)

        # init model
        self.model = RobertaForMultiLabelClassification(
            model_name="roberta-large",
            num_labels=config["num_labels"],
            dropout_rate=config.get("dropout_rate", 0.3),
            use_mean_pooling=config.get("use_mean_pooling", True)
        )

        # load weights
        weights_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        # load thresholds (optional use)
        thresholds_path = hf_hub_download(repo_id=model_name, filename="optimal_thresholds.json")
        with open(thresholds_path) as f:
            self.thresholds = np.array(json.load(f))

        # labels
        self.labels = [
            'admiration','amusement','anger','annoyance','approval','caring','confusion',
            'curiosity','desire','disappointment','disapproval','disgust','embarrassment',
            'excitement','fear','gratitude','grief','joy','love','nervousness','optimism',
            'pride','realization','relief','remorse','sadness','surprise','neutral'
        ]

        print("RoBERTa model loaded successfully")

    def detectEmotion(self, text):

        if not text.strip():
            return [("neutral", 100.0)]

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

        probs = torch.sigmoid(logits).cpu().numpy()[0]

        emotionScores = []

        for i in range(len(probs)):
            emotionScores.append((self.labels[i], probs[i] * 100))

        emotionScores.sort(key=lambda x: x[1], reverse=True)

        return emotionScores[:3]