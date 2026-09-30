"""Data loader for enVENT and EmoWOZ datasets.
Calculates Eq. (3) ground-truth appraisal targets (d, p, rho, q, mask),
creates splits, and filters classes according to Table III.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_DIR, ENVENT_COVERED_LABELS, EMOWOZ_COVERED_LABELS


def compute_envent_targets(row):
    """Compute (d, p, rho, q, mask) according to Eq. (3) of the paper."""
    unit = lambda x: (float(x) - 1.0) / 4.0        # [1,5] -> [0,1]
    bipolar = lambda x: (float(x) - 3.0) / 2.0     # [1,5] -> [-1,1]

    pl = unit(row["pleasantness"])
    un = unit(row["unpleasantness"])
    goal = bipolar(row["goal_support"])
    d = 0.5 * (pl - un + goal)
    d = np.clip(d, -1.0, 1.0)

    std = unit(row["standards"])
    norm = unit(row["social_norms"])
    b = 0.5 * (std + norm)

    own = float(row["self_responsblt"])
    other = float(row["other_responsblt"])
    sit = float(row["chance_responsblt"])

    rho = max(0.0, 0.5 * (max(own, other) - 3.0))
    p = (1.0 - b) * rho * max(0.0, d) - b
    p = np.clip(p, -1.0, 1.0)

    mask = 1.0 if max(own, other) >= 3.0 else 0.0

    raw_q = np.array([own - 1.0, other - 1.0, sit - 1.0], dtype=float)
    if raw_q.sum() > 0:
        q = raw_q / raw_q.sum()
    else:
        q = np.full(3, 1.0 / 3.0)

    return float(d), float(p), float(mask), q.tolist()


def load_envent():
    """Load enVENT corpus, compute appraisal targets, and create splits."""
    gen_file = DATA_DIR / "envent" / "corpus" / "crowd-enVent_generation.tsv"
    val_file = DATA_DIR / "envent" / "corpus" / "crowd-enVent_validation.tsv"

    if not gen_file.exists():
        raise FileNotFoundError(f"enVENT generation file not found at {gen_file}")

    df_gen = pd.read_csv(gen_file, sep="\t")
    df_val = pd.read_csv(val_file, sep="\t")
    test_text_ids = set(df_val["text_id"].unique())

    # Map raw emotions
    def map_emotion(emo):
        if emo in ("guilt", "shame"):
            return "guilt/shame"
        return emo

    df_gen["clean_emotion"] = df_gen["emotion"].apply(map_emotion)

    # Compute targets for each row
    targets = [compute_envent_targets(r) for _, r in df_gen.iterrows()]
    df_gen["target_d"] = [t[0] for t in targets]
    df_gen["target_p"] = [t[1] for t in targets]
    df_gen["target_mask"] = [t[2] for t in targets]
    df_gen["target_q"] = [t[3] for t in targets]
    df_gen["target_agent"] = [int(np.argmax(t[3])) for t in targets]

    # Test split: 1200 validated texts
    df_test = df_gen[df_gen["text_id"].isin(test_text_ids)].copy().reset_index(drop=True)
    df_rest = df_gen[~df_gen["text_id"].isin(test_text_ids)].copy().reset_index(drop=True)

    # Train / Val split: 4860 train / 540 val (stratified by clean_emotion)
    df_train, df_val_split = train_test_split(
        df_rest, test_size=540, random_state=42, stratify=df_rest["clean_emotion"]
    )
    df_train = df_train.reset_index(drop=True)
    df_val_split = df_val_split.reset_index(drop=True)

    # Classification subsets (only the 6 covered labels)
    df_train_clf = df_train[df_train["clean_emotion"].isin(ENVENT_COVERED_LABELS)].reset_index(drop=True)
    df_val_clf = df_val_split[df_val_split["clean_emotion"].isin(ENVENT_COVERED_LABELS)].reset_index(drop=True)
    df_test_clf = df_test[df_test["clean_emotion"].isin(ENVENT_COVERED_LABELS)].reset_index(drop=True)

    print(f"[enVENT] Loaded {len(df_gen)} total rows:")
    print(f"  Train: {len(df_train)} (Clf: {len(df_train_clf)})")
    print(f"  Val:   {len(df_val_split)} (Clf: {len(df_val_clf)})")
    print(f"  Test:  {len(df_test)} (Clf: {len(df_test_clf)})")
    print(f"  Test label distribution: {dict(df_test_clf['clean_emotion'].value_counts())}")

    return {
        "full": {"train": df_train, "val": df_val_split, "test": df_test},
        "clf": {"train": df_train_clf, "val": df_val_clf, "test": df_test_clf},
    }


def load_emowoz():
    """Load EmoWOZ multiwoz dialogue turns for cross-domain evaluation."""
    split_file = DATA_DIR / "emowoz" / "data-split.json"
    multiwoz_file = DATA_DIR / "emowoz" / "emowoz-multiwoz.json"

    if not split_file.exists() or not multiwoz_file.exists():
        raise FileNotFoundError(f"EmoWOZ files not found at {DATA_DIR / 'emowoz'}")

    with open(split_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    with open(multiwoz_file, "r", encoding="utf-8") as f:
        dialogues = json.load(f)

    # Map from EmoWOZ integer codes:
    # 0: neutral, 1: fearful, 2: dissatisfied, 3: apologetic, 4: abusive, 5: excited, 6: satisfied
    INT_TO_EMO = {
        0: "neutral",
        1: "fearful",
        2: "dissatisfied/abusive",
        3: "apologetic",
        4: "dissatisfied/abusive",
        5: "excited",
        6: "satisfied",
    }

    # Collect test turns from test dialogues
    test_dial_ids = set(splits.get("test", {}).get("multiwoz", []))
    turns = []

    for dial_id in test_dial_ids:
        if dial_id not in dialogues:
            continue
        dial = dialogues[dial_id]
        log = dial.get("log", [])
        prev_sys = ""
        for i, turn in enumerate(log):
            text = turn.get("text", "")
            # Even turns are user turns, odd turns are system turns
            if i % 2 == 1:
                prev_sys = text
                continue

            # User turn
            emotion_info = turn.get("emotion", [])
            # Consensus label is in turn['emotion'][3]['emotion'] if available
            if emotion_info and len(emotion_info) >= 4 and "emotion" in emotion_info[3]:
                emo_code = emotion_info[3]["emotion"]
            elif emotion_info and "annotation" in emotion_info[0]:
                emo_code = emotion_info[0]["annotation"]
            else:
                emo_code = 0

            clean_emo = INT_TO_EMO.get(emo_code, "neutral")

            # Elicitor / Agent mapping according to OCC and EmoWOZ definitions:
            # - dissatisfied/abusive, satisfied: operator conduct -> Other (1)
            # - apologetic: user conduct -> Self (0)
            # - fearful: event/circumstance -> Circumstance (2)
            # - excited: positive outcome -> Circumstance (2) or Self (0)
            # - neutral: Circumstance (2)
            if clean_emo in ("dissatisfied/abusive", "satisfied"):
                agent_idx = 1
                agent_name = "Other"
            elif clean_emo == "apologetic":
                agent_idx = 0
                agent_name = "Self"
            elif clean_emo == "fearful":
                agent_idx = 2
                agent_name = "Circumstance"
            elif clean_emo == "excited":
                agent_idx = 2
                agent_name = "Circumstance"
            else:
                agent_idx = 2
                agent_name = "Circumstance"

            # Valence: +1 for positive, -1 for negative, 0 for neutral
            if clean_emo in ("excited", "satisfied"):
                valence = 1
            elif clean_emo in ("fearful", "dissatisfied/abusive", "apologetic"):
                valence = -1
            else:
                valence = 0

            turns.append({
                "dialogue_id": dial_id,
                "turn_id": i,
                "text": text,
                "context": prev_sys,
                "emotion": clean_emo,
                "agent_idx": agent_idx,
                "agent_name": agent_name,
                "valence": valence,
            })
            prev_sys = text

    df_test = pd.DataFrame(turns)
    print(f"[EmoWOZ] Loaded {len(df_test)} test turns from {len(test_dial_ids)} test dialogues.")
    print(f"  Label distribution:\n{df_test['emotion'].value_counts()}")
    return df_test


if __name__ == "__main__":
    envent_data = load_envent()
    emowoz_data = load_emowoz()
    print("All data loaded and validated successfully!")
