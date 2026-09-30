"""Module 1: Deep Neural Cognitive Appraisal Extractor.
Implements RoBERTa-large encoder with:
1) Multi-task regression head for (d, p) in [-1, 1]^2 via GELU and Tanh.
2) Classification head for agent distribution pi in Delta^2 via Softmax.
3) Combined loss according to Eq. (4) of the paper.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class AppraisalExtractor(nn.Module):
    def __init__(self, model_name="roberta-large", hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        encoder_dim = self.encoder.config.hidden_size

        # Appraisal head (d, p) with GELU and Tanh (Eq. 1)
        self.appraisal_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
            nn.Tanh()
        )

        # Agent head for 3 classes: (Self, Other, Circumstance) (Eq. 2)
        self.agent_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, 3)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        # Extract representation of the initial token (<s> / [CLS])
        h = outputs.last_hidden_state[:, 0, :]

        # Predict (d, p) in [-1, 1]^2
        dp_pred = self.appraisal_head(h)  # [B, 2]
        d_pred = dp_pred[:, 0]
        p_pred = dp_pred[:, 1]

        # Predict agent logits in R^3
        agent_logits = self.agent_head(h)  # [B, 3]
        agent_probs = torch.softmax(agent_logits, dim=-1)

        return d_pred, p_pred, agent_probs, agent_logits

    def compute_loss(self, d_pred, p_pred, agent_logits, target_d, target_p, target_m, target_q, lambda_agent=1.0):
        """Compute multi-task loss according to Eq. (4) of the paper.
        target_d: [B]
        target_p: [B]
        target_m: [B] (mask for p)
        target_q: [B, 3] (soft target distribution for agents)
        """
        # Desirability loss (MSE)
        loss_d = torch.mean((d_pred - target_d) ** 2)

        # Praiseworthiness loss with mask m_i (MSE)
        masked_p_diff = target_m * ((p_pred - target_p) ** 2)
        if target_m.sum() > 0:
            loss_p = masked_p_diff.sum() / target_m.sum()
        else:
            loss_p = torch.tensor(0.0, device=d_pred.device)

        # Agent distribution loss: Cross-Entropy with soft probabilities target_q
        # - sum_a q_{i,a} * log(pi_{i,a})
        log_probs = torch.log_softmax(agent_logits, dim=-1)
        loss_agent = -torch.mean(torch.sum(target_q * log_probs, dim=-1))

        total_loss = loss_d + loss_p + lambda_agent * loss_agent
        return total_loss, loss_d, loss_p, loss_agent
