"""Training script for Module 1 (Neural Cognitive Appraisal Extractor).
Trains RoBERTa-large on enVENT appraisal targets using Eq. (4) multi-task loss.
Evaluates on validation split, performs early stopping, saves best checkpoint,
and generates out-of-sample predictions for enVENT test and EmoWOZ test.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config import (
    BATCH_SIZE,
    DEVICE,
    ENCODER_NAME,
    LEARNING_RATE,
    MAX_EPOCHS,
    MODELS_DIR,
    PATIENCE,
    RESULTS_DIR,
    WEIGHT_DECAY,
    LAMBDA_AGENT,
)
from data_loader import load_envent, load_emowoz
from module1_model import AppraisalExtractor


class AppraisalDataset(Dataset):
    def __init__(self, texts, contexts=None, targets=None, tokenizer=None, max_length=128):
        self.texts = list(texts)
        self.contexts = list(contexts) if contexts is not None else [""] * len(self.texts)
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        ctx = str(self.contexts[idx])

        # Sentence pair encoding (Text U and context C)
        if ctx and ctx.strip():
            encoding = self.tokenizer(
                text,
                ctx,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        else:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        if self.targets is not None:
            item["target_d"] = torch.tensor(self.targets["d"][idx], dtype=torch.float32)
            item["target_p"] = torch.tensor(self.targets["p"][idx], dtype=torch.float32)
            item["target_m"] = torch.tensor(self.targets["m"][idx], dtype=torch.float32)
            item["target_q"] = torch.tensor(self.targets["q"][idx], dtype=torch.float32)

        return item


def train_epoch(model, dataloader, optimizer, scheduler, scaler):
    model.train()
    total_loss, total_loss_d, total_loss_p, total_loss_agent = 0.0, 0.0, 0.0, 0.0

    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        token_type_ids = batch.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(DEVICE)

        target_d = batch["target_d"].to(DEVICE)
        target_p = batch["target_p"].to(DEVICE)
        target_m = batch["target_m"].to(DEVICE)
        target_q = batch["target_q"].to(DEVICE)

        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            d_pred, p_pred, agent_probs, agent_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            loss, loss_d, loss_p, loss_agent = model.compute_loss(
                d_pred, p_pred, agent_logits,
                target_d, target_p, target_m, target_q,
                lambda_agent=LAMBDA_AGENT
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        total_loss_d += loss_d.item()
        total_loss_p += loss_p.item()
        total_loss_agent += loss_agent.item()

    n = len(dataloader)
    return total_loss / n, total_loss_d / n, total_loss_p / n, total_loss_agent / n


def evaluate(model, dataloader):
    model.eval()
    total_loss, total_loss_d, total_loss_p, total_loss_agent = 0.0, 0.0, 0.0, 0.0
    all_d_preds, all_p_preds, all_agent_preds = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(DEVICE)

            target_d = batch["target_d"].to(DEVICE)
            target_p = batch["target_p"].to(DEVICE)
            target_m = batch["target_m"].to(DEVICE)
            target_q = batch["target_q"].to(DEVICE)

            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                d_pred, p_pred, agent_probs, agent_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                loss, loss_d, loss_p, loss_agent = model.compute_loss(
                    d_pred, p_pred, agent_logits,
                    target_d, target_p, target_m, target_q,
                    lambda_agent=LAMBDA_AGENT
                )

            total_loss += loss.item()
            total_loss_d += loss_d.item()
            total_loss_p += loss_p.item()
            total_loss_agent += loss_agent.item()

            all_d_preds.extend(d_pred.cpu().numpy())
            all_p_preds.extend(p_pred.cpu().numpy())
            all_agent_preds.extend(agent_probs.cpu().numpy())

    n = len(dataloader)
    return (
        total_loss / n,
        total_loss_d / n,
        total_loss_p / n,
        total_loss_agent / n,
        np.array(all_d_preds),
        np.array(all_p_preds),
        np.array(all_agent_preds),
    )


def predict(model, dataloader):
    model.eval()
    all_d_preds, all_p_preds, all_agent_preds = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(DEVICE)

            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                d_pred, p_pred, agent_probs, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
            all_d_preds.extend(d_pred.cpu().numpy())
            all_p_preds.extend(p_pred.cpu().numpy())
            all_agent_preds.extend(agent_probs.cpu().numpy())

    return np.array(all_d_preds), np.array(all_p_preds), np.array(all_agent_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--model-name", type=str, default=ENCODER_NAME)
    parser.add_argument("--skip-train-if-exists", action="store_true")
    args = parser.parse_args()

    checkpoint_path = MODELS_DIR / "module1_roberta_occ.pt"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 1. Load data
    print("Loading enVENT data...")
    envent = load_envent()
    train_df = envent["full"]["train"]
    val_df = envent["full"]["val"]
    test_df = envent["full"]["test"]

    train_targets = {
        "d": train_df["target_d"].values,
        "p": train_df["target_p"].values,
        "m": train_df["target_mask"].values,
        "q": np.stack(train_df["target_q"].values),
    }
    val_targets = {
        "d": val_df["target_d"].values,
        "p": val_df["target_p"].values,
        "m": val_df["target_mask"].values,
        "q": np.stack(val_df["target_q"].values),
    }

    train_ds = AppraisalDataset(train_df["generated_text"], targets=train_targets, tokenizer=tokenizer)
    val_ds = AppraisalDataset(val_df["generated_text"], targets=val_targets, tokenizer=tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = AppraisalExtractor(model_name=args.model_name).to(DEVICE)

    if args.skip_train_if_exists and checkpoint_path.exists():
        print(f"Loading existing checkpoint from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    else:
        print(f"Training Module 1 on {DEVICE} for {args.epochs} epochs...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            train_loss, train_d, train_p, train_a = train_epoch(model, train_loader, optimizer, scheduler, scaler)
            val_loss, val_d, val_p, val_a, _, _, _ = evaluate(model, val_loader)

            print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} (d:{train_d:.3f}, p:{train_p:.3f}, a:{train_a:.3f}) | Val Loss: {val_loss:.4f} (d:{val_d:.3f}, p:{val_p:.3f}, a:{val_a:.3f})")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  --> Saved best model checkpoint to {checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        # Load best model
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    # 2. Predict on enVENT test set
    print("\nGenerating Module 1 predictions on enVENT test set (1200 texts)...")
    test_ds = AppraisalDataset(test_df["generated_text"], tokenizer=tokenizer)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    d_test, p_test, agent_test = predict(model, test_loader)

    np.savez_compressed(
        RESULTS_DIR / "envent_module1_preds.npz",
        d=d_test,
        p=p_test,
        pi=agent_test,
        text_id=test_df["text_id"].values,
        clean_emotion=test_df["clean_emotion"].values,
        target_d=test_df["target_d"].values,
        target_p=test_df["target_p"].values,
        target_mask=test_df["target_mask"].values,
        target_agent=test_df["target_agent"].values,
        target_q=np.stack(test_df["target_q"].values),
    )
    print(f"Saved enVENT predictions to {RESULTS_DIR / 'envent_module1_preds.npz'}")

    # 3. Predict on EmoWOZ test set
    print("\nGenerating Module 1 predictions on EmoWOZ test set...")
    emowoz_df = load_emowoz()
    emowoz_ds = AppraisalDataset(emowoz_df["text"], contexts=emowoz_df["context"], tokenizer=tokenizer)
    emowoz_loader = DataLoader(emowoz_ds, batch_size=args.batch_size, shuffle=False)
    d_emowoz, p_emowoz, agent_emowoz = predict(model, emowoz_loader)

    np.savez_compressed(
        RESULTS_DIR / "emowoz_module1_preds.npz",
        d=d_emowoz,
        p=p_emowoz,
        pi=agent_emowoz,
        emotion=emowoz_df["emotion"].values,
        valence=emowoz_df["valence"].values,
        agent_idx=emowoz_df["agent_idx"].values,
        agent_name=emowoz_df["agent_name"].values,
    )
    print(f"Saved EmoWOZ predictions to {RESULTS_DIR / 'emowoz_module1_preds.npz'}")
    print("\nModule 1 training and inference complete!")


if __name__ == "__main__":
    main()
