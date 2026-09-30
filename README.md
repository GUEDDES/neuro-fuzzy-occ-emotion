# From Affect to Cognition: A Neuro-Fuzzy Emotion Recognition Framework Based on the OCC Appraisal Model

Official implementation and experimental reproduction repository for the IEEE conference paper:
> **"From Affect to Cognition: A Neuro-Fuzzy Emotion Recognition Framework Based on the OCC Appraisal Model"**  
> *Afef Ben Said, Abdelwaheb Gueddes, Imen Jegham, Mohamed Ali Mahjoub*  
> LATIS Laboratory, ENISO / ISITC, Sousse University, Tunisia.

---

## 📌 Overview

Traditional text emotion recognition systems rely either on direct black-box classification (e.g., RoBERTa fine-tuned on discrete classes) or continuous affective dimensional mapping (Valence-Arousal-Dominance - VAD). While dimensional spaces describe *what* an individual feels, they fail to model the *cognitive etiology*—the evaluation that caused the feeling. Consequently, emotions that feel alike but stem from distinct causes (such as **sadness**, **anger at a wrongdoer**, and **guilt**) are frequently conflated.

This framework introduces a **Neuro-Symbolic Cognitive Appraisal Architecture**:
1. **Module 1 (Deep Neural Perception)**: A RoBERTa-large encoder trained on cognitive appraisal ratings from the **crowd-enVENT** corpus to predict continuous event Desirability ($\hat{d} \in [-1, 1]$), action Praiseworthiness ($\hat{p} \in [-1, 1]$), and a probability distribution over the responsible agent ($\boldsymbol{\pi} \in \Delta^2$: *Self, Other, Circumstance*). **Emotion labels are never used during Module 1 training.**
2. **Module 2 (Fuzzy Symbolic OCC Engine)**: A Mamdani Fuzzy Inference System (FIS) utilizing 5-term Ruspini partitions, soft agent memberships ($\mu_a = \pi_a$), and 52 formal Ortony-Clore-Collins (OCC) psychological rules to deduce compound OCC emotion types, activation strengths ($\alpha_e$), and defuzzified centroid intensities ($I_e \in [0, 1]$).
3. **Explainable AI (XAI)**: Generates faithful, human-readable rationales directly identifying the triggered rule, linguistic terms, and agent attribution.

```
                      +-------------------------------------------------+
                      |             USER UTTERANCE (U, C)               |
                      +-----------------------+-------------------------+
                                              |
                                              v
                      +-------------------------------------------------+
                      |       MODULE 1: NEURAL APPRAISAL EXTRACTOR      |
                      |          (RoBERTa-large Multi-Task)             |
                      |  • Desirability: d in [-1, 1]                   |
                      |  • Praiseworthiness: p in [-1, 1]               |
                      |  • Responsible Agent: pi in Delta^2             |
                      +-----------------------+-------------------------+
                                              |
                                              v
                      +-------------------------------------------------+
                      |        MODULE 2: FUZZY SYMBOLIC OCC ENGINE      |
                      |  • 5-Term Ruspini Partitions (HU, U, N, D, HD)  |
                      |  • 52 OCC Psychological Appraisal Rules         |
                      |  • Min T-Norm & Max Aggregation                 |
                      |  • Decoupled Activation (alpha) & Centroid (I)  |
                      +-----------------------+-------------------------+
                                              |
                                              v
                      +-------------------------------------------------+
                      |               DECISION & XAI OUTPUT             |
                      |  • Dominant OCC Emotion & Target Dataset Label  |
                      |  • Defuzzified Intensity (Low, Medium, High)    |
                      |  • Faithful Rule-Derived Explanation            |
                      +-------------------------------------------------+
```

---

## 🚀 Quick Start & Interactive Demo

Test the trained model on custom sentences with real-time explainability:

```bash
# Single sentence inference
python demo_inference.py --text "The driver took a wrong turn on purpose, and I missed my sister's wedding."

# Interactive command-line loop
python demo_inference.py --interactive
```

### Example Output:
```
======================================================================
INPUT TEXT: "The driver took a wrong turn on purpose, and I missed my sister's wedding."
----------------------------------------------------------------------
1. NEURAL COGNITIVE EXTRACTION (Module 1):
   * Event Desirability (d)    : -0.88  [-1.0: highly undesirable, +1.0: highly desirable]
   * Action Praiseworthiness (p): -0.78  [-1.0: highly blameworthy,   +1.0: highly praiseworthy]
   * Responsible Agent (pi)    :
     - Self        :  1.9%  [                    ]
     - Other       : 79.5%  [###############     ]
     - Circumstance: 18.6%  [###                 ]

2. FUZZY SYMBOLIC OCC INFERENCE (Module 2):
   * Primary OCC Emotion       : Anger
   * Rule Activation (alpha)   : 0.80
   * Centroid Intensity (I)    : 0.77 (High intensity)
   * enVENT Projected Label    : ANGER
   * EmoWOZ Dialogue Label     : DISSATISFIED/ABUSIVE

3. EXPLAINABLE AI (XAI) RATIONALE:
   "Anger (activation 0.80, High intensity): the event is evaluated as HU (1.00), other is responsible (prob 0.80) and the action is evaluated as HB (0.93)."
======================================================================
```

---

## 📦 Installation & Environment

### Prerequisites
* Python 3.10 or 3.11
* NVIDIA GPU with CUDA support (e.g., RTX 4050, RTX 3060, or better)
* Git & Git LFS

### Setup
```bash
# Clone the repository
git clone https://github.com/GUEDDES/neuro-fuzzy-occ-emotion.git
cd neuro-fuzzy-occ-emotion

# Install dependencies
pip install -r requirements.txt
```

---

## 🛠️ Complete Experimental Workflow

### Step 1: Download & Preprocess Datasets
Downloads **crowd-enVENT** and **EmoWOZ**, computes ground-truth appraisal targets $(d_i, p_i, \boldsymbol{q}_i)$ according to Eq. (3), and creates data splits:
```bash
python src/data_loader.py
```

### Step 2: Train Module 1 (Appraisal Extractor)
Trains the multi-task RoBERTa-large model using the combined loss function (Eq. 4). Mixed precision (AMP fp16) and early stopping are enabled:
```bash
python src/train_module1.py --epochs 5 --batch-size 16 --lr 2e-5
```
*Outputs: Checkpoint saved to `models/module1_roberta_occ.pt` and test predictions to `results/`.*

### Step 3: Run Full Benchmark & Evaluation
Runs the complete experimental matrix across all baselines and ablations:
* **B1**: Direct fine-tuned classifier (RoBERTa-large)
* **B2**: VAD-T1FIS & CAT2-NFI (Dimensional affective baseline)
* **B3**: VAD Probe (Linear regression on 3 VAD features)
* **B4**: OCC Probe (Linear regression on 5 OCC features)
* **Ours**: Proposed OCC Neuro-Fuzzy Framework
* **A1**: Crisp boolean rule ablation
* **A2**: Hard agent decision ablation

```bash
python src/run_experiments.py
```
*Outputs: Complete metrics saved to `results/all_metrics.json`.*

### Step 4: Update LaTeX Paper & Compile PDF
Injects measured metrics into `occ_fuzzy_recognition_conference.tex` and compiles the submission-ready PDF:
```bash
python src/update_latex.py
```
*Outputs: `occ_fuzzy_recognition_conference.pdf` (7 pages IEEE format).*

---

## 📊 Summary of Experimental Results

### RQ1: Appraisal Estimation Quality (Module 1)
| Dataset | Target Variable | Metric | Empirical Value |
| :--- | :--- | :--- | :---: |
| **enVENT (test)** | Desirability $\hat{d}$ | Pearson $r$ / RMSE | **0.81 / 0.44** |
| | Praiseworthiness $\hat{p}$ | Pearson $r$ / RMSE | **0.74 / 0.39** |
| | Responsible Agent ($\boldsymbol{\pi}$) | Macro-F1 | **69.7%** |
| **EmoWOZ (dialogue)** | Valence transfer (sign of $\hat{d}$) | Accuracy | **91.9%** |
| | Dialogue Elicitor alignment | Macro-F1 | **35.7%** |

### RQ2 & RQ3: Emotion Recognition & Same-Valence Confusion (SVC)
| System | Emotion Representation | enVENT Macro-F1 | enVENT Accuracy | enVENT SVC ($\downarrow$) | EmoWOZ Macro-F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **B1 Direct classifier** | Supervised End-to-End | 76.9% | 77.0% | 17.3% | 0.4% |
| **B2 VAD-T1FIS** | Dimensional Affect (VAD) | 27.3% | 28.7% | 52.9% | 12.6% |
| **B2 CAT2-NFI (IT2)** | Interval Type-2 VAD | 27.3% | 28.7% | 52.9% | 12.6% |
| **B3 VAD probe** | Linear Probe on VAD | 37.5% | 46.5% | 40.2% | 3.8% |
| **B4 OCC probe** | Linear Probe on OCC | 59.2% | 59.7% | 27.5% | 3.5% |
| **Ours (OCC-T1FIS)** | **Cognitive Appraisal (OCC Rules)** | **54.5%** | **54.8%** | **33.8%** | **9.8%** |
| **A1 Crisp inference** | Boolean Rules Ablation | 55.0% | 55.2% | 32.8% | 8.5% |
| **A2 Hard agent** | Hard Agent Indicator Ablation | 54.4% | 54.7% | 33.8% | 9.4% |

**Key Findings:**
1. **OCC vs. VAD**: Under identical fuzzy inference mechanisms, **Ours (+27.2% Macro-F1)** significantly outperforms VAD-T1FIS ($p = 1.4 \times 10^{-19}$), while reducing same-valence confusion by nearly 20 points.
2. **Unsupervised Emotion Mapping**: Without ever training on emotion labels, our rule-based system reaches **54.5% Macro-F1**, closely approaching the supervised probe **B4 (59.2%)** while offering 100% faithful explanations.
3. **Intensity & XAI (RQ4)**: Defuzzified centroid intensity $I_{\hat{e}}$ correlates with self-reported writer intensity (Spearman $\rho = 0.38$, $p < 0.001$). Explanations achieved 92% human plausibility agreement ($\kappa = 0.79$).

---

## 📂 Repository Structure

```
.
├── src/
│   ├── config.py                 # Global paths, hyperparameters, Ruspini MFs, OCC tables
│   ├── data_loader.py            # Automated download & parsing for enVENT and EmoWOZ
│   ├── module1_model.py          # RoBERTa-large multi-task appraisal architecture
│   ├── train_module1.py          # Training loop with AMP fp16 and early stopping
│   ├── fuzzy_engine.py           # Vectorized Mamdani FIS with 52 OCC rules and XAI
│   ├── baselines.py              # Implementation of B1-B4 baselines and A1-A2 ablations
│   ├── evaluate.py               # Evaluation metrics, bootstrap CIs, McNemar tests, SVC
│   ├── run_experiments.py        # Master benchmark orchestrator
│   └── update_latex.py           # LaTeX updater & PDF compiler
├── models/
│   └── module1_roberta_occ.pt    # Trained PyTorch checkpoint (tracked via Git LFS)
├── results/
│   ├── all_metrics.json          # Consolidated empirical results for all experiments
│   ├── envent_module1_preds.npz  # Out-of-sample predictions on enVENT test split
│   └── emowoz_module1_preds.npz  # Zero-shot predictions on EmoWOZ test split
├── demo_inference.py             # CLI & interactive demonstration script
├── requirements.txt              # Python library dependencies
├── occ_fuzzy_recognition_conference.tex  # Camera-ready IEEE LaTeX source
├── occ_fuzzy_recognition_conference.pdf  # Final compiled 7-page IEEE paper
└── README.md                     # Comprehensive documentation
```

---

## 📄 Citation

```bibtex
@inproceedings{bensaid2026cognition,
  author    = {Ben Said, Afef and Gueddes, Abdelwaheb and Jegham, Imen and Mahjoub, Mohamed Ali},
  title     = {From Affect to Cognition: A Neuro-Fuzzy Emotion Recognition Framework Based on the OCC Appraisal Model},
  booktitle = {IEEE Conference on Affective Computing and Intelligent Interaction},
  year      = {2026},
  address   = {Sousse, Tunisia}
}
```
