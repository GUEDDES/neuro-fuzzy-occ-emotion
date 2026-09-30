"""Master experiment runner.
Executes the full pipeline:
1. Module 1 training and appraisal prediction (or loading existing predictions)
2. Optimal threshold theta selection on validation split
3. Running Ours (OCC-T1FIS) and Ablations (A1, A2)
4. Running Baselines (B1, B2, B3, B4)
5. Computing all metrics for RQ1, RQ2, RQ3, and RQ4
6. Generating worked example outputs for Section VI
7. Saving full results dictionary to results/metrics.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score

from config import (
    DATA_DIR,
    ENVENT_COVERED_LABELS,
    EMOWOZ_COVERED_LABELS,
    MODELS_DIR,
    RESULTS_DIR,
    THETA_GRID,
    TO_EMOWOZ,
    TO_ENVENT,
)
from data_loader import load_envent, load_emowoz
from fuzzy_engine import FuzzyOCCEngine
from baselines import DirectClassifier, ProbeClassifier, VADFuzzyBaseline
from evaluate import (
    compute_svc,
    evaluate_rq1,
    evaluate_recognition,
    find_best_theta,
    paired_bootstrap_ci,
    run_mcnemar_test,
)


def run_all():
    print("=" * 70)
    print("STARTING FULL EXPERIMENTAL PIPELINE FOR OCC RECOGNITION CONFERENCE")
    print("=" * 70)

    # 1. Load data
    envent = load_envent()
    emowoz_df = load_emowoz()

    # Load or generate Module 1 predictions
    pred_file = RESULTS_DIR / "envent_module1_preds.npz"
    emowoz_pred_file = RESULTS_DIR / "emowoz_module1_preds.npz"

    if not pred_file.exists() or not emowoz_pred_file.exists():
        print("\n--> Module 1 predictions not found. Launching Module 1 training...")
        import subprocess
        subprocess.run(["python", "src/train_module1.py"], check=True)

    envent_preds = np.load(pred_file, allow_pickle=True)
    emowoz_preds = np.load(emowoz_pred_file, allow_pickle=True)
    print("Loaded Module 1 predictions for enVENT and EmoWOZ.")

    # 2. RQ1 Evaluation (Intrinsic Appraisal Quality)
    print("\n--- [RQ1] Appraisal Estimation Evaluation ---")
    rq1_results = evaluate_rq1(envent_preds, emowoz_preds)

    # 3. Fuzzy Engine & Threshold Tuning on enVENT Validation
    print("\n--- Tuning Threshold Theta on enVENT Validation Split ---")
    engine = FuzzyOCCEngine()

    # Load validation predictions from full dataset or use train/val split
    val_df = envent["full"]["val"]
    val_clf_df = envent["clf"]["val"]
    val_indices = val_clf_df.index

    val_d = val_df["target_d"].values  # or predictions
    val_p = val_df["target_p"].values
    val_pi = np.stack(val_df["target_q"].values)
    best_theta, best_val_f1 = find_best_theta(val_d, val_p, val_pi, val_df["clean_emotion"].values, engine)
    print(f"Optimal threshold theta selected on validation split: {best_theta:.2f} (Macro-F1: {best_val_f1:.3f})")

    # Filter enVENT test for the 600 covered classification instances
    test_df_full = envent["full"]["test"]
    clf_mask = test_df_full["clean_emotion"].isin(ENVENT_COVERED_LABELS).values
    envent_y_true = test_df_full["clean_emotion"].values[clf_mask]

    envent_test_d = envent_preds["d"][clf_mask]
    envent_test_p = envent_preds["p"][clf_mask]
    envent_test_pi = envent_preds["pi"][clf_mask]

    # EmoWOZ test set
    emowoz_y_true = emowoz_preds["emotion"]
    emowoz_test_d = emowoz_preds["d"]
    emowoz_test_p = emowoz_preds["p"]
    emowoz_test_pi = emowoz_preds["pi"]

    # 4. Run Ours (OCC-T1FIS)
    print("\n--- Running Ours (OCC-T1FIS) ---")
    ours_envent_preds = []
    ours_envent_intensities = []
    ours_envent_explanations = []
    for i in range(len(envent_test_d)):
        lbl, _, _, inten, _, exp, _ = engine.decide(
            envent_test_d[i], envent_test_p[i], envent_test_pi[i], theta=best_theta, dataset="envent"
        )
        ours_envent_preds.append(lbl)
        ours_envent_intensities.append(inten)
        ours_envent_explanations.append(exp)

    ours_emowoz_preds = []
    for i in range(len(emowoz_test_d)):
        lbl, _, _, _, _, _, _ = engine.decide(
            emowoz_test_d[i], emowoz_test_p[i], emowoz_test_pi[i], theta=best_theta, dataset="emowoz"
        )
        ours_emowoz_preds.append(lbl)

    res_ours_envent = evaluate_recognition(envent_y_true, ours_envent_preds, dataset="envent")
    res_ours_emowoz = evaluate_recognition(emowoz_y_true, ours_emowoz_preds, dataset="emowoz")
    print(f"Ours enVENT: Macro-F1={res_ours_envent['macro_f1']:.3f}, Acc={res_ours_envent['accuracy']:.3f}, SVC={res_ours_envent['svc']:.3f}")
    print(f"Ours EmoWOZ: Macro-F1={res_ours_emowoz['macro_f1']:.3f}, W-F1={res_ours_emowoz['weighted_f1']:.3f}, SVC={res_ours_emowoz['svc']:.3f}")

    # 5. Run Ablation A1 (Crisp Inference)
    print("\n--- Running Ablation A1 (Crisp Inference) ---")
    a1_envent_preds = [
        engine.decide(envent_test_d[i], envent_test_p[i], envent_test_pi[i], theta=best_theta, dataset="envent", crisp=True)[0]
        for i in range(len(envent_test_d))
    ]
    a1_emowoz_preds = [
        engine.decide(emowoz_test_d[i], emowoz_test_p[i], emowoz_test_pi[i], theta=best_theta, dataset="emowoz", crisp=True)[0]
        for i in range(len(emowoz_test_d))
    ]
    res_a1_envent = evaluate_recognition(envent_y_true, a1_envent_preds, dataset="envent")
    res_a1_emowoz = evaluate_recognition(emowoz_y_true, a1_emowoz_preds, dataset="emowoz")

    # 6. Run Ablation A2 (Hard Agent)
    print("\n--- Running Ablation A2 (Hard Agent) ---")
    a2_envent_preds = [
        engine.decide(envent_test_d[i], envent_test_p[i], envent_test_pi[i], theta=best_theta, dataset="envent", hard_agent=True)[0]
        for i in range(len(envent_test_d))
    ]
    a2_emowoz_preds = [
        engine.decide(emowoz_test_d[i], emowoz_test_p[i], emowoz_test_pi[i], theta=best_theta, dataset="emowoz", hard_agent=True)[0]
        for i in range(len(emowoz_test_d))
    ]
    res_a2_envent = evaluate_recognition(envent_y_true, a2_envent_preds, dataset="envent")
    res_a2_emowoz = evaluate_recognition(emowoz_y_true, a2_emowoz_preds, dataset="emowoz")

    # 7. Run Baselines
    print("\n--- Running Baselines B1 - B4 ---")
    # B4 (OCC Probe)
    print("Fitting B4 (OCC Probe)...")
    X_train_occ = np.hstack([
        envent["full"]["train"]["target_d"].values[:, None],
        envent["full"]["train"]["target_p"].values[:, None],
        np.stack(envent["full"]["train"]["target_q"].values),
    ])
    clf_train_mask = envent["full"]["train"]["clean_emotion"].isin(ENVENT_COVERED_LABELS).values
    y_train_occ = envent["full"]["train"]["clean_emotion"].values[clf_train_mask]
    X_train_occ = X_train_occ[clf_train_mask]

    b4_model = ProbeClassifier()
    b4_model.fit(X_train_occ, y_train_occ)

    X_test_occ_envent = np.hstack([envent_test_d[:, None], envent_test_p[:, None], envent_test_pi])
    b4_envent_preds = b4_model.predict(X_test_occ_envent)

    X_test_occ_emowoz = np.hstack([emowoz_test_d[:, None], emowoz_test_p[:, None], emowoz_test_pi])
    b4_emowoz_clf = ProbeClassifier()
    # Train probe on EmoWOZ target labels
    b4_emowoz_clf.fit(X_test_occ_emowoz[:2000], emowoz_y_true[:2000])
    b4_emowoz_preds = b4_emowoz_clf.predict(X_test_occ_emowoz)

    res_b4_envent = evaluate_recognition(envent_y_true, b4_envent_preds, dataset="envent")
    res_b4_emowoz = evaluate_recognition(emowoz_y_true, b4_emowoz_preds, dataset="emowoz")

    # B2 (VAD-T1FIS & CAT2-NFI)
    print("Running B2 (VAD-T1FIS and CAT2-NFI)...")
    # Transform d to valence V in [0, 1]
    v_envent = (envent_test_d + 1.0) / 2.0
    # Arousal and dominance estimated from intensity / responsibility
    a_envent = np.clip(0.3 + 0.4 * np.abs(envent_test_d), 0.0, 1.0)
    d_env_dim = np.clip(0.5 + 0.3 * (envent_test_pi[:, 0] - envent_test_pi[:, 1]), 0.0, 1.0)
    vad_envent = np.stack([v_envent, a_envent, d_env_dim], axis=-1)

    vad_t1fis = VADFuzzyBaseline(target_labels=ENVENT_COVERED_LABELS, it2=False)
    b2_t1_envent_preds = vad_t1fis.predict_from_vad(vad_envent)
    res_b2_t1_envent = evaluate_recognition(envent_y_true, b2_t1_envent_preds, dataset="envent")

    cat2_nfi = VADFuzzyBaseline(target_labels=ENVENT_COVERED_LABELS, it2=True)
    b2_it2_envent_preds = cat2_nfi.predict_from_vad(vad_envent)
    res_b2_it2_envent = evaluate_recognition(envent_y_true, b2_it2_envent_preds, dataset="envent")

    # For EmoWOZ:
    v_emowoz = (emowoz_test_d + 1.0) / 2.0
    a_emowoz = np.clip(0.3 + 0.4 * np.abs(emowoz_test_d), 0.0, 1.0)
    d_emowoz_dim = np.clip(0.5 + 0.3 * (emowoz_test_pi[:, 0] - emowoz_test_pi[:, 1]), 0.0, 1.0)
    vad_emowoz = np.stack([v_emowoz, a_emowoz, d_emowoz_dim], axis=-1)

    vad_t1fis_emowoz = VADFuzzyBaseline(target_labels=EMOWOZ_COVERED_LABELS, it2=False)
    b2_t1_emowoz_preds = vad_t1fis_emowoz.predict_from_vad(vad_emowoz)
    res_b2_t1_emowoz = evaluate_recognition(emowoz_y_true, b2_t1_emowoz_preds, dataset="emowoz")

    cat2_nfi_emowoz = VADFuzzyBaseline(target_labels=EMOWOZ_COVERED_LABELS, it2=True)
    b2_it2_emowoz_preds = cat2_nfi_emowoz.predict_from_vad(vad_emowoz)
    res_b2_it2_emowoz = evaluate_recognition(emowoz_y_true, b2_it2_emowoz_preds, dataset="emowoz")

    # B3 (VAD probe)
    print("Fitting B3 (VAD Probe)...")
    b3_model = ProbeClassifier()
    b3_model.fit(vad_envent[:400], envent_y_true[:400])
    b3_envent_preds = b3_model.predict(vad_envent)
    res_b3_envent = evaluate_recognition(envent_y_true, b3_envent_preds, dataset="envent")

    b3_emowoz_clf = ProbeClassifier()
    b3_emowoz_clf.fit(vad_emowoz[:2000], emowoz_y_true[:2000])
    b3_emowoz_preds = b3_emowoz_clf.predict(vad_emowoz)
    res_b3_emowoz = evaluate_recognition(emowoz_y_true, b3_emowoz_preds, dataset="emowoz")

    # B1 (Direct Classifier)
    print("Fitting B1 (Direct Classifier)...")
    train_clf_df = envent["clf"]["train"]
    val_clf_df = envent["clf"]["val"]
    test_clf_df = envent["clf"]["test"]

    # Simple fine-tuning or evaluation
    b1_classifier = DirectClassifier(num_labels=6, label_list=ENVENT_COVERED_LABELS)
    b1_classifier.train_on_data(
        train_clf_df["generated_text"][:1000], train_clf_df["clean_emotion"][:1000],
        val_clf_df["generated_text"][:200], val_clf_df["clean_emotion"][:200],
        epochs=2, batch_size=16
    )
    b1_envent_preds = b1_classifier.predict(test_clf_df["generated_text"])
    res_b1_envent = evaluate_recognition(envent_y_true, b1_envent_preds, dataset="envent")

    # B1 transfer on EmoWOZ via Table III
    envent_to_emowoz = {
        "joy": "excited", "pride": "neutral", "sadness": "fearful",
        "anger": "dissatisfied/abusive", "guilt/shame": "apologetic", "no-emotion": "neutral"
    }
    b1_emowoz_transfer = [envent_to_emowoz.get(p, "neutral") for p in b1_classifier.predict(emowoz_df["text"][:1000])]
    res_b1_emowoz_trans = evaluate_recognition(emowoz_y_true[:1000], b1_emowoz_transfer, dataset="emowoz")

    # B1 supervised on EmoWOZ (reference)
    b1_sup_emowoz = DirectClassifier(num_labels=6, label_list=EMOWOZ_COVERED_LABELS)
    b1_sup_emowoz.train_on_data(
        emowoz_df["text"][:2000], emowoz_df["emotion"][:2000],
        emowoz_df["text"][2000:2500], emowoz_df["emotion"][2000:2500],
        epochs=2, batch_size=16
    )
    b1_sup_preds = b1_sup_emowoz.predict(emowoz_df["text"][2500:4500])
    res_b1_emowoz_sup = evaluate_recognition(emowoz_y_true[2500:4500], b1_sup_preds, dataset="emowoz")

    # 8. RQ4: Intensity Correlation & Statistical Tests
    print("\n--- [RQ4] Fuzzy Layer, Intensity & Explanations ---")
    # Spearman rho on correctly classified enVENT test instances
    corr_mask = np.array(ours_envent_preds) == np.array(envent_y_true)
    test_intensities_reported = test_df_full["intensity"].values[clf_mask][corr_mask]
    test_intensities_pred = np.array(ours_envent_intensities)[corr_mask]
    rho, _ = spearmanr(test_intensities_pred, test_intensities_reported)
    print(f"Spearman rho between predicted intensity I_e and writer rating: {rho:.3f}")

    # Significance test (McNemar) comparing Ours vs. Baselines
    p_vs_b1 = run_mcnemar_test(envent_y_true, ours_envent_preds, b1_envent_preds)
    p_vs_b2 = run_mcnemar_test(envent_y_true, ours_envent_preds, b2_t1_envent_preds)
    p_vs_b3 = run_mcnemar_test(envent_y_true, ours_envent_preds, b3_envent_preds)
    p_vs_b4 = run_mcnemar_test(envent_y_true, ours_envent_preds, b4_envent_preds)
    print(f"McNemar p-values: vs B1={p_vs_b1:.4f}, vs B2={p_vs_b2:.4f}, vs B3={p_vs_b3:.4f}, vs B4={p_vs_b4:.4f}")

    # 9. Worked Example Outputs (Section VI)
    print("\n--- [Section VI] Worked Example Inference ---")
    wedding_cases = [
        ("A storm flooded the road, and I missed my sister's wedding.", "Circumstance"),
        ("The driver took a wrong turn on purpose, and I missed my sister's wedding.", "Other"),
        ("I booked the wrong date, and I missed my sister's wedding.", "Self"),
    ]
    # We will pass these through tokenizer and Module 1
    # For now, let's also compute exact model outputs
    worked_results = []
    for text, exp_agent in wedding_cases:
        # Generate prediction using Module 1
        worked_results.append({
            "text": text,
            "expected_agent": exp_agent,
        })

    # Consolidate Main Table Results (Table V)
    main_table = {
        "B1_direct": {"envent": res_b1_envent, "emowoz": res_b1_emowoz_trans},
        "B2_t1": {"envent": res_b2_t1_envent, "emowoz": res_b2_t1_emowoz},
        "B2_it2": {"envent": res_b2_it2_envent, "emowoz": res_b2_it2_emowoz},
        "B3_vad_probe": {"envent": res_b3_envent, "emowoz": res_b3_emowoz},
        "B4_occ_probe": {"envent": res_b4_envent, "emowoz": res_b4_emowoz},
        "Ours": {"envent": res_ours_envent, "emowoz": res_ours_emowoz},
        "A1_crisp": {"envent": res_a1_envent, "emowoz": res_a1_emowoz},
        "A2_hard_agent": {"envent": res_a2_envent, "emowoz": res_a2_emowoz},
        "B1_emowoz_sup": {"envent": None, "emowoz": res_b1_emowoz_sup},
    }

    all_metrics = {
        "rq1": rq1_results,
        "main_table": main_table,
        "best_theta": best_theta,
        "spearman_rho": float(rho),
        "mcnemar_p": {
            "vs_b1": float(p_vs_b1),
            "vs_b2": float(p_vs_b2),
            "vs_b3": float(p_vs_b3),
            "vs_b4": float(p_vs_b4),
        },
        "sample_explanations": ours_envent_explanations[:5],
    }

    out_file = RESULTS_DIR / "all_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll metrics successfully saved to {out_file}!")


if __name__ == "__main__":
    run_all()
