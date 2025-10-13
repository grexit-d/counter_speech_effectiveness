import torch
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Learnable Dependency Matrix
class LearnableDependencyMatrix(nn.Module):
    def __init__(self, num_tasks):
        super(LearnableDependencyMatrix, self).__init__()
        self.matrix = nn.Parameter(torch.rand(num_tasks, num_tasks))

    def forward(self):
        matrix = self.matrix.clone()
        for i in range(matrix.size(0)):
            matrix[i, i] = 0.0
        return matrix


# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce_loss = self.bce_loss(logits, targets)
        probs = torch.sigmoid(logits)
        focal_loss = self.alpha * (1 - probs) ** self.gamma * bce_loss
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()


# Dataset Class
class MultiTaskDataset(Dataset):
    def __init__(
        self,
        input_ids,
        attention_mask,
        labels_emotional,
        labels_audience,
        labels_clarity,
        labels_evidence,
        labels_rebuttal,
        labels_fairness,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels_emotional = labels_emotional
        self.labels_audience = labels_audience
        self.labels_clarity = labels_clarity
        self.labels_evidence = labels_evidence
        self.labels_rebuttal = labels_rebuttal
        self.labels_fairness = labels_fairness

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels_emotional": self.labels_emotional[idx],
            "labels_audience": self.labels_audience[idx],
            "labels_clarity": self.labels_clarity[idx],
            "labels_evidence": self.labels_evidence[idx],
            "labels_rebuttal": self.labels_rebuttal[idx],
            "labels_fairness": self.labels_fairness[idx],
        }


# Preprocessing Function
def preprocess_data(df, tokenizer, max_length=128, only_cs=False):
    df = df.dropna(
        subset=[
            "counter_speech",
            "hate_speech",
            "clarity",
            "evidence",
            "rebuttal",
            "fairness",
            "emotional_appeal",
            "audience_adaptation",
        ]
    )
    df["emotional_appeal"] = df["emotional_appeal"].astype(int)
    df["audience_adaptation"] = df["audience_adaptation"].astype(int)
    df["clarity"] = df["clarity"].astype(int) - 1
    df["evidence"] = df["evidence"].astype(int) - 1
    df["rebuttal"] = df["rebuttal"].astype(int) - 1
    df["fairness"] = df["fairness"].astype(int) - 1

    if only_cs:
        text = df["counter_speech"]
    else:
        text = df["hate_speech"] + " " + df["counter_speech"]

    encodings = tokenizer(
        list(text),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    labels_emotional = torch.tensor(
        df["emotional_appeal"].values, dtype=torch.float32
    ).unsqueeze(1)
    labels_audience = torch.tensor(
        df["audience_adaptation"].values, dtype=torch.float32
    ).unsqueeze(1)
    labels_clarity = torch.tensor(df["clarity"].values, dtype=torch.long)
    labels_evidence = torch.tensor(df["evidence"].values, dtype=torch.long)
    labels_rebuttal = torch.tensor(df["rebuttal"].values, dtype=torch.long)
    labels_fairness = torch.tensor(df["fairness"].values, dtype=torch.long)

    return (
        encodings["input_ids"],
        encodings["attention_mask"],
        labels_emotional,
        labels_audience,
        labels_clarity,
        labels_evidence,
        labels_rebuttal,
        labels_fairness,
    )
