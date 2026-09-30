"""Implementation of baselines and ablations:
- B1: Direct fine-tuned RoBERTa classifier
- B2: VAD-T1FIS (Type-1 FIS using NRC-VAD prototypes) & CAT2-NFI
- B3: VAD Probe (logistic regression on 3 VAD features)
- B4: OCC Probe (logistic regression on 5 OCC features)
- A1: Crisp Inference ablation
- A2: Hard Agent ablation
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    DEVICE,
    ENCODER_NAME,
    ENVENT_COVERED_LABELS,
    EMOWOZ_COVERED_LABELS,
    MODELS_DIR,
    TO_EMOWOZ,
    TO_ENVENT,
)
from fuzzy_engine import FuzzyOCCEngine

# NRC-VAD Lexicon class prototype centroids on [0, 1]^3 (Mohammad, 2018)
# Matching CAT2-NFI (Ben Said et al., 2024)
NRC_VAD_PROTOTYPES = {
    # enVENT labels
    "joy": np.array([0.980, 0.767, 0.796]),
    "pride": np.array([0.880, 0.640, 0.840]),
    "sadness": np.array([0.100, 0.250, 0.150]),
    "anger": np.array([0.167, 0.865, 0.651]),
    "guilt/shame": np.array([0.130, 0.550, 0.250]),
    "no-emotion": np.array([0.500, 0.300, 0.500]),

    # EmoWOZ labels
    "excited": np.array([0.950, 0.850, 0.750]),
    "satisfied": np.array([0.850, 0.450, 0.700]),
    "fearful": np.array([0.150, 0.800, 0.200]),
    "dissatisfied/abusive": np.array([0.150, 0.850, 0.600]),
    "apologetic": np.array([0.250, 0.450, 0.250]),
    "neutral": np.array([0.500, 0.300, 0.500]),
}


class DirectClassifier:
    """B1: RoBERTa-large direct sequence classification model."""
    def __init__(self, num_labels=6, model_name=ENCODER_NAME, label_list=ENVENT_COVERED_LABELS):
        self.num_labels = num_labels
        self.label_list = label_list
        self.label2id = {l: i for i, l in enumerate(label_list)}
        self.id2label = {i: l for i, l in enumerate(label_list)}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(DEVICE)

    def train_on_data(self, train_texts, train_labels, val_texts, val_labels, epochs=3, batch_size=16, lr=2e-5):
        train_y = [self.label2id[l] for l in train_labels]
        val_y = [self.label2id[l] for l in val_labels]

        train_enc = self.tokenizer(list(train_texts), padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        val_enc = self.tokenizer(list(val_texts), padding="max_length", max_length=128, truncation=True, return_tensors="pt")

        train_dataset = torch.utils.data.TensorDataset(train_enc["input_ids"], train_enc["attention_mask"], torch.tensor(train_y))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

        self.model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                b_ids, b_mask, b_y = [x.to(DEVICE) for x in batch]
                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    out = self.model(input_ids=b_ids, attention_mask=b_mask, labels=b_y)
                    loss = out.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

    def predict(self, texts, batch_size=32):
        self.model.eval()
        preds = []
        enc = self.tokenizer(list(texts), padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        dataset = torch.utils.data.TensorDataset(enc["input_ids"], enc["attention_mask"])
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch in loader:
                b_ids, b_mask = [x.to(DEVICE) for x in batch]
                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    out = self.model(input_ids=b_ids, attention_mask=b_mask)
                    logits = out.logits
                    p = torch.argmax(logits, dim=-1).cpu().numpy()
                    preds.extend(p)

        return [self.id2label[i] for i in preds]


class VADFuzzyBaseline:
    """B2: VAD-T1FIS and CAT2-NFI (Type-1 / Type-2 Fuzzy System on VAD space)."""
    def __init__(self, target_labels=ENVENT_COVERED_LABELS, it2=False):
        self.target_labels = target_labels
        self.it2 = it2
        self.prototypes = {l: NRC_VAD_PROTOTYPES[l] for l in target_labels if l in NRC_VAD_PROTOTYPES}

    def predict_from_vad(self, vad_coords, uncertainty=None):
        """vad_coords: [N, 3] in [0, 1]^3.
        Computes membership to class prototypes using Gaussian / Euclidean fuzzy kernel.
        """
        preds = []
        for i in range(len(vad_coords)):
            v = vad_coords[i]
            best_label = None
            best_score = -1e9
            for label, proto in self.prototypes.items():
                dist = np.linalg.norm(v - proto)
                sigma = 0.25 if not self.it2 else (0.22 if uncertainty is None else 0.25 - 0.05 * uncertainty[i])
                score = np.exp(-(dist ** 2) / (2 * (sigma ** 2)))
                if score > best_score:
                    best_score = score
                    best_label = label
            preds.append(best_label)
        return preds


class ProbeClassifier:
    """B3 (VAD probe) and B4 (OCC probe) using Multinomial Logistic Regression."""
    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs")

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        return self.clf.predict(X_test)


def run_fuzzy_pipeline(d_preds, p_preds, agent_preds, dataset="envent", theta=0.3, crisp=False, hard_agent=False):
    """Run FuzzyOCCEngine on batch of predictions."""
    engine = FuzzyOCCEngine()
    preds = []
    intensities = []
    explanations = []

    for i in range(len(d_preds)):
        d = float(d_preds[i])
        p = float(p_preds[i])
        pi = agent_preds[i]

        label, best_e, alpha, intensity, term, exp, profile = engine.decide(
            d=d, p=p, pi=pi, theta=theta, dataset=dataset, crisp=crisp, hard_agent=hard_agent
        )
        preds.append(label)
        intensities.append(intensity)
        explanations.append(exp)

    return preds, intensities, explanations
