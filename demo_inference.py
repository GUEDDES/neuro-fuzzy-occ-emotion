"""Interactive and CLI Demo for Neuro-Fuzzy OCC Emotion Recognition.

Usage:
    python demo_inference.py --text "The driver took a wrong turn on purpose, and I missed my sister's wedding."
    python demo_inference.py --interactive
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import AGENTS, DEVICE, ENCODER_NAME, MODELS_DIR
from fuzzy_engine import FuzzyOCCEngine
from module1_model import AppraisalExtractor


def load_framework(model_path=None):
    if model_path is None:
        model_path = MODELS_DIR / "module1_roberta_occ.pt"

    if not model_path.exists():
        print(f"Error: Model weights not found at {model_path}.")
        print("Please train the model first by running: python src/train_module1.py")
        sys.exit(1)

    print(f"Loading RoBERTa-large encoder and appraisal heads from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    model = AppraisalExtractor(model_name=ENCODER_NAME).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    engine = FuzzyOCCEngine()
    return tokenizer, model, engine


def analyze_text(text, context="", tokenizer=None, model=None, engine=None, theta=0.35):
    if context and context.strip():
        enc = tokenizer(text, context, return_tensors="pt", max_length=128, padding=True, truncation=True).to(DEVICE)
    else:
        enc = tokenizer(text, return_tensors="pt", max_length=128, padding=True, truncation=True).to(DEVICE)

    with torch.no_grad():
        d_pred, p_pred, agent_probs, _ = model(enc["input_ids"], enc["attention_mask"])

    d = float(d_pred.cpu().numpy()[0])
    p = float(p_pred.cpu().numpy()[0])
    pi = agent_probs.cpu().numpy()[0]
    pi_dict = dict(zip(AGENTS, [float(x) for x in pi]))

    label_envent, occ_e, alpha, intensity, term, exp_envent, profile = engine.decide(
        d=d, p=p, pi=pi_dict, theta=theta, dataset="envent"
    )
    label_emowoz = engine.decide(d=d, p=p, pi=pi_dict, theta=theta, dataset="emowoz")[0]

    return {
        "text": text,
        "context": context,
        "desirability": d,
        "praiseworthiness": p,
        "agent_distribution": pi_dict,
        "dominant_occ_emotion": occ_e,
        "activation": alpha,
        "intensity": intensity,
        "intensity_term": term,
        "envent_label": label_envent,
        "emowoz_label": label_emowoz,
        "explanation": exp_envent,
        "all_activations": {k: v[0] for k, v in profile.items()},
    }


def print_report(res):
    print("\n" + "=" * 70)
    print(f"INPUT TEXT: \"{res['text']}\"")
    if res['context']:
        print(f"CONTEXT:    \"{res['context']}\"")
    print("-" * 70)
    print("1. NEURAL COGNITIVE EXTRACTION (Module 1):")
    print(f"   * Event Desirability (d)    : {res['desirability']:+.2f}  [-1.0: highly undesirable, +1.0: highly desirable]")
    print(f"   * Action Praiseworthiness (p): {res['praiseworthiness']:+.2f}  [-1.0: highly blameworthy,   +1.0: highly praiseworthy]")
    print("   * Responsible Agent (pi)    :")
    for a, prob in res['agent_distribution'].items():
        bar = "#" * int(prob * 20)
        print(f"     - {a:12s}: {prob:5.1%}  [{bar:<20s}]")
    print("\n2. FUZZY SYMBOLIC OCC INFERENCE (Module 2):")
    print(f"   * Primary OCC Emotion       : {res['dominant_occ_emotion']}")
    print(f"   * Rule Activation (alpha)   : {res['activation']:.2f}")
    print(f"   * Centroid Intensity (I)    : {res['intensity']:.2f} ({res['intensity_term']} intensity)")
    print(f"   * enVENT Projected Label    : {res['envent_label'].upper()}")
    print(f"   * EmoWOZ Dialogue Label     : {res['emowoz_label'].upper()}")
    print("\n3. EXPLAINABLE AI (XAI) RATIONALE:")
    print(f"   \"{res['explanation']}\"")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Neuro-Fuzzy OCC Emotion Recognition Demo")
    parser.add_argument("--text", type=str, default=None, help="Text to analyze")
    parser.add_argument("--context", type=str, default="", help="Optional preceding context turn")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive command-line loop")
    parser.add_argument("--theta", type=float, default=0.35, help="Fuzzy activation threshold")
    args = parser.parse_args()

    tokenizer, model, engine = load_framework()

    if args.interactive or args.text is None:
        print("\n--- Neuro-Fuzzy OCC Interactive Emotion Analysis ---")
        print("Type an utterance and press Enter (or 'quit' to exit).\n")
        while True:
            try:
                user_text = input("Enter text: ").strip()
                if not user_text:
                    continue
                if user_text.lower() in ("quit", "exit", "q"):
                    break
                res = analyze_text(user_text, tokenizer=tokenizer, model=model, engine=engine, theta=args.theta)
                print_report(res)
            except KeyboardInterrupt:
                break
    else:
        res = analyze_text(args.text, context=args.context, tokenizer=tokenizer, model=model, engine=engine, theta=args.theta)
        print_report(res)


if __name__ == "__main__":
    main()
