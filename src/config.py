"""Configuration file for the Neuro-Fuzzy OCC emotion recognition framework.
"""

from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Trapezoid parameters (a1, a2, a3, a4); triangles have a2 == a3. (Table I of the paper)
INPUT_TERMS = {
    "--": (-1.0, -1.0, -0.8, -0.5),  # HU / HB
    "-": (-0.8, -0.5, -0.5, -0.2),   # U  / B
    "0": (-0.5, -0.2, 0.2, 0.5),     # N
    "+": (0.2, 0.5, 0.5, 0.8),       # D  / P
    "++": (0.5, 0.8, 1.0, 1.0),      # HD / HP
}

# Calibrated Ruspini partition parameters tuned on enVENT validation split
INPUT_TERMS_CALIBRATED = {
    "--": (-1.0, -1.0, -0.80, -0.55),
    "-": (-0.80, -0.55, -0.55, -0.25),
    "0": (-0.55, -0.25, 0.25, 0.55),
    "+": (0.25, 0.55, 0.55, 0.80),
    "++": (0.55, 0.80, 1.0, 1.0),
}

# Calibrated OCC rule weights tuned on enVENT validation split
CALIBRATED_WEIGHTS = {
    "Joy": 1.325,
    "Gratitude": 1.325,
    "Admiration": 1.325,
    "Pride": 0.884,
    "Gratification": 0.884,
    "Distress": 1.551,
    "Anger": 0.790,
    "Reproach": 0.790,
    "Shame": 1.216,
    "Remorse": 1.216,
}
W_NEUTRAL = 1.075
IT2_DELTA = 0.03

D_NAMES = {"--": "HU", "-": "U", "0": "N", "+": "D", "++": "HD"}
P_NAMES = {"--": "HB", "-": "B", "0": "N", "+": "P", "++": "HP"}

OUTPUT_TERMS = {
    "Low": (0.0, 0.0, 0.2, 0.5),
    "Medium": (0.2, 0.5, 0.5, 0.8),
    "High": (0.5, 0.8, 1.0, 1.0),
}

AGENTS = ("Self", "Other", "Circumstance")
COMPOUND = {"Gratification", "Remorse", "Gratitude", "Anger"}

# Appraisal variables each OCC type depends on
DEPENDS = {
    "Joy": {"d"}, "Distress": {"d"},
    "Pride": {"p"}, "Shame": {"p"}, "Admiration": {"p"}, "Reproach": {"p"},
    "Gratification": {"d", "p"}, "Remorse": {"d", "p"},
    "Gratitude": {"d", "p"}, "Anger": {"d", "p"},
}

# Table II of the paper: (valence of d, valence of p) -> consequent types.
GRID = {
    "Self": {
        ("-", "-"): ["Remorse"], ("-", "0"): ["Distress"], ("-", "+"): ["Distress", "Pride"],
        ("0", "-"): ["Shame"], ("0", "+"): ["Pride"],
        ("+", "-"): ["Joy", "Shame"], ("+", "0"): ["Joy"], ("+", "+"): ["Gratification"],
    },
    "Other": {
        ("-", "-"): ["Anger"], ("-", "0"): ["Distress"], ("-", "+"): ["Distress", "Admiration"],
        ("0", "-"): ["Reproach"], ("0", "+"): ["Admiration"],
        ("+", "-"): ["Joy", "Reproach"], ("+", "0"): ["Joy"], ("+", "+"): ["Gratitude"],
    },
}
VALENCE_TERMS = {"-": ["--", "-"], "0": ["0"], "+": ["+", "++"]}
EXTREME = {"--", "++"}

# Table III of the paper: Mapping of OCC types to dataset labels
TO_ENVENT = {
    "Joy": "joy",
    "Gratitude": "joy",
    "Admiration": "joy",
    "Pride": "pride",
    "Gratification": "pride",
    "Distress": "sadness",
    "Anger": "anger",
    "Reproach": "anger",
    "Shame": "guilt/shame",
    "Remorse": "guilt/shame",
    None: "no-emotion",
}

TO_EMOWOZ = {
    "Joy": "excited",
    "Gratitude": "satisfied",
    "Admiration": "satisfied",
    "Pride": "(no class)",
    "Gratification": "(no class)",
    "Distress": "fearful",
    "Anger": "dissatisfied/abusive",
    "Reproach": "dissatisfied/abusive",
    "Shame": "apologetic",
    "Remorse": "apologetic",
    None: "neutral",
}

ENVENT_COVERED_LABELS = ["joy", "pride", "sadness", "anger", "guilt/shame", "no-emotion"]
EMOWOZ_COVERED_LABELS = ["excited", "satisfied", "fearful", "dissatisfied/abusive", "apologetic", "neutral"]

# Valence groups for SVC (Same-Valence Confusion) calculation (Section IV-C)
SVC_GROUPS_ENVENT = {
    "negative": ["sadness", "anger", "guilt/shame"],
    "positive": ["joy", "pride"],
}
SVC_GROUPS_EMOWOZ = {
    "negative": ["fearful", "dissatisfied/abusive", "apologetic"],
    "positive": ["excited", "satisfied"],
}

# Training Hyperparameters
ENCODER_NAME = "roberta-large"
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 8
PATIENCE = 3
LAMBDA_AGENT = 1.0
THETA_GRID = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
