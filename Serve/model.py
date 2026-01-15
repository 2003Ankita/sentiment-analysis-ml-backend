import torch.nn as nn
from transformers import AutoModel

class TransformerSentimentClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", dropout=0.2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 1)  # binary

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]         # CLS token
        logits = self.classifier(self.dropout(cls))  # (B,1)
        return logits.squeeze(-1)                    # (B,)
