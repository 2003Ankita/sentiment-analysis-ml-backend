import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

from model import TransformerSentimentClassifier

class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }

def evaluate(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)

            y_true.extend(labels.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    auc = None
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred)
    return acc, p, r, f1, auc, cm

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model_artifacts'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', './data_bert'))

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.model_dir, exist_ok=True)

    train_path = os.path.join(args.data_dir, "train_text.csv")
    val_path = os.path.join(args.data_dir, "val_text.csv")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_ds = TextDataset(train_df, tokenizer, max_len=args.max_len)
    val_ds = TextDataset(val_df, tokenizer, max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = TransformerSentimentClassifier(model_name=args.model_name).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        acc, p, r, f1, auc, cm = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch} | loss={total_loss/len(train_loader):.4f} | "
            f"acc={acc:.4f} p={p:.4f} r={r:.4f} f1={f1:.4f} auc={auc} | cm=\n{cm}"
        )

    # Save artifacts for serving
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pt"))

    tokenizer_dir = os.path.join(args.model_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_dir)

    with open(os.path.join(args.model_dir, "model_meta.json"), "w") as f:
        json.dump({"model_name": args.model_name, "max_len": args.max_len}, f)

    print("Saved artifacts to:", args.model_dir)

if __name__ == "__main__":
    main()
