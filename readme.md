# Fraud Detection — A Step-by-Step Teaching Demo

A classroom-friendly Python script that compares a **single model** against **bagging** and **boosting** on a fraud-detection problem. It runs in six clearly labelled steps and prints *what* it is doing and *why* at each stage, so students can follow the whole story in the console.

---

## What this demo teaches

- Why fraud detection is hard (the data is heavily imbalanced).
- Why **accuracy is misleading** and what to use instead.
- How **bagging** and **boosting** improve on a single model.
- How to read **precision, recall, F1, and ROC-AUC**.

---

## Requirements

```bash
pip install scikit-learn matplotlib
```

Run it with:

```bash
python fraud_teaching.py
```

---

## The six steps (what & why)

### Step 0 — The problem
Sets the scene: a payments company wants to catch fraud without raising too many false alarms on genuine customers. Fraud is rare, which makes it tricky.

### Step 1 — Create the data
**What:** generate 20,000 example transactions, each labelled fraud (1) or genuine (0), with only ~1.5% fraud.
**Why:** we need labelled examples to learn from. The script highlights the key trap here — because fraud is so rare, a lazy model that calls *everything* genuine would score ~98% accuracy while catching zero fraud. That's why we don't trust accuracy.

### Step 2 — Split into train & test
**What:** keep 70% for training, 30% for testing, using `stratify` to preserve the fraud ratio in both halves.
**Why:** we must measure performance on data the model has **never seen**, otherwise we're just testing its memory. Stratifying keeps the test fair (the test set still has the same tiny fraud percentage).

### Step 3 — The three models
- **Single tree** (before) — one decision tree, our baseline.
- **Bagging** (after) — 200 trees averaged; very stable, cuts **variance**.
- **Boosting** (after) — 200 trees in a chain, each fixing the previous one's mistakes; cuts **bias**, focuses on the hard fraud cases.

### Step 4 — Train each model and measure it
**What:** for each model: train it, predict on the test set, and measure the result.
**Why:** training teaches the model the patterns; predicting on unseen data shows real performance. Each metric is printed *with its meaning* next to it so the numbers aren't abstract.

### Step 5 — Compare side by side
A summary table of all three models plus a plain-language takeaway: both ensembles beat the single tree; bagging is the most precise (fewest false alarms); boosting has the best overall ranking.

### Step 6 — Show the chart
Builds and saves `comparison.png` at the very end, so students see the bars side by side after reading the narrative.

---

## The code

```python
"""
Fraud detection teaching demo: Single model vs Bagging vs Boosting
Runs step by step and explains WHAT we do and WHY at each stage.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

RANDOM_STATE = 42


def banner(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


banner("STEP 0  —  THE PROBLEM")
print(
    "We run a payments company. A few transactions are FRAUD, most are\n"
    "genuine. We want a model that catches fraud WITHOUT annoying real\n"
    "customers with false alarms. Fraud is very rare, so this is tricky."
)

banner("STEP 1  —  CREATE THE DATA")
print("WHY: we need example transactions, each labelled fraud (1) or genuine (0).")
X, y = make_classification(
    n_samples=20000, n_features=12, n_informative=8,
    weights=[0.99, 0.01], class_sep=1.3, random_state=RANDOM_STATE,
)
n_fraud = int(y.sum())
print(f"\nTotal transactions : {len(y):,}")
print(f"Genuine            : {len(y) - n_fraud:,}")
print(f"Fraud              : {n_fraud:,}  (only {y.mean()*100:.2f}% of all data!)")
print("\nNOTE: because fraud is so rare, 'accuracy' is a trap — a lazy model that")
print("calls everything 'genuine' would be ~98% accurate but catch ZERO fraud.")

banner("STEP 2  —  SPLIT INTO TRAIN & TEST")
print("WHY: we train on one part, then test on data the model has never seen,")
print("to measure REAL performance (not memorisation). 'stratify' keeps the")
print("same fraud ratio in both halves so the test is fair.")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
print(f"\nTraining transactions : {len(y_train):,}")
print(f"Testing transactions  : {len(y_test):,}  (fraud in test: {int(y_test.sum())})")

banner("STEP 3  —  THE THREE MODELS WE COMPARE")
print("BEFORE -> Single tree : one model, our baseline.")
print("AFTER  -> Bagging     : 200 trees averaged  (cuts variance, very stable).")
print("AFTER  -> Boosting    : 200 trees in a chain, each fixing the last's")
print("                        mistakes (cuts bias, focuses on hard fraud).")
models = {
    "Single tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
    "Bagging":     RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                          random_state=RANDOM_STATE, n_jobs=-1),
    "Boosting":    GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

banner("STEP 4  —  TRAIN EACH MODEL & MEASURE IT")
print("For each model we: (a) train it, (b) predict on the test set,")
print("(c) measure how well it found fraud.\n")

results = {}
for name, model in models.items():
    print("-" * 64)
    print(f">> {name}: training on {len(y_train):,} transactions...")
    model.fit(X_train, y_train)
    print("   training done. Now predicting on unseen test data...")

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    results[name] = {"precision": precision, "recall": recall, "f1": f1, "auc": auc}

    print(f"   Caught {tp} of {tp+fn} real frauds, with {fp} false alarms.")
    print(f"   Precision = {precision:.2f}  (of flagged, how many were truly fraud)")
    print(f"   Recall    = {recall:.2f}  (of all fraud, how much we caught)")
    print(f"   F1        = {f1:.2f}  (balance of precision & recall)")
    print(f"   ROC-AUC   = {auc:.2f}  (how well it ranks fraud above genuine)")

banner("STEP 5  —  COMPARE THEM SIDE BY SIDE")
print(f"{'Model':14}{'Precision':>11}{'Recall':>9}{'F1':>7}{'ROC-AUC':>9}")
print("-" * 50)
for name, m in results.items():
    print(f"{name:14}{m['precision']:>11.2f}{m['recall']:>9.2f}{m['f1']:>7.2f}{m['auc']:>9.2f}")
print("-" * 50)
print("Takeaway: both ensembles beat the single tree. Bagging is the most")
print("PRECISE (fewest false alarms); Boosting has the best overall ranking")
print("(highest ROC-AUC). The single tree is the noisiest, least reliable one.")

banner("STEP 6  —  SHOW THE CHART (saved as comparison.png)")
metrics = ["precision", "recall", "f1", "auc"]
nice = ["Precision", "Recall", "F1", "ROC-AUC"]
labels = list(results.keys())
colors = ["#888780", "#1D9E75", "#7F77DD"]
x = np.arange(len(metrics))
width = 0.25
plt.figure(figsize=(9, 5))
for i, name in enumerate(labels):
    plt.bar(x + i * width, [results[name][k] for k in metrics], width,
            label=name, color=colors[i])
plt.xticks(x + width, nice)
plt.ylabel("Score (higher is better)")
plt.title("Single model vs Bagging vs Boosting")
plt.legend()
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("comparison.png", dpi=130)
print("Chart saved. Open comparison.png to see the bars side by side.")
```

---

## Example console output

```
================================================================
STEP 1  —  CREATE THE DATA
================================================================
Total transactions : 20,000
Genuine            : 19,695
Fraud              : 305  (only 1.52% of all data!)

================================================================
STEP 4  —  TRAIN EACH MODEL & MEASURE IT
================================================================
>> Single tree: ...
   Caught 27 of 91 real frauds, with 70 false alarms.
   Precision = 0.28   Recall = 0.30   F1 = 0.29   ROC-AUC = 0.64
>> Bagging: ...
   Caught 19 of 91 real frauds, with 0 false alarms.
   Precision = 1.00   Recall = 0.21   F1 = 0.35   ROC-AUC = 0.77
>> Boosting: ...
   Caught 26 of 91 real frauds, with 36 false alarms.
   Precision = 0.42   Recall = 0.29   F1 = 0.34   ROC-AUC = 0.79
```

| Model | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|------|---------|
| Single tree | 0.28 | 0.30 | 0.29 | 0.64 |
| Bagging | 1.00 | 0.21 | 0.35 | 0.77 |
| Boosting | 0.42 | 0.29 | 0.34 | 0.79 |

*(Numbers vary slightly with parameters, but the pattern is stable.)*

## Quick metric reminders

- **Precision** — of what we flagged as fraud, how much really was fraud (fewer false alarms).
- **Recall** — of all the real fraud, how much we caught (less slips through).
- **F1** — a single score balancing precision and recall.
- **ROC-AUC** — how well the model ranks fraud above genuine; 0.5 = guessing, 1.0 = perfect.
- **Why not accuracy?** — with ~1.5% fraud, calling everything "genuine" scores ~98% accuracy and catches nothing. Useless here.

---

## Terminologies

Plain-language explanations of every term used above.

### The building blocks

Every metric is built from four counts:

- **True Positive (TP)** — a fraud the model correctly flagged as fraud. ✅
- **False Positive (FP)** — a genuine transaction wrongly flagged as fraud (a *false alarm*). ❌
- **False Negative (FN)** — a fraud the model missed and let through. ❌
- **True Negative (TN)** — a genuine transaction correctly let through. ✅

### Precision
*Of everything the model flagged as fraud, how much was really fraud?*

```
Precision = TP / (TP + FP)
```

High precision = few false alarms. Precision of 1.00 means everything flagged truly was fraud.

### Recall
*Of all the fraud that actually existed, how much did the model catch?*

```
Recall = TP / (TP + FN)
```

High recall = little fraud slips through. Recall of 0.30 means we caught only 30% of real fraud.

### Difference between Precision and Recall

They sound alike but answer two **different** questions and divide by two **different** things:

| | Precision | Recall |
|---|-----------|--------|
| **Question** | When I say "fraud", am I right? | Did I catch all the fraud? |
| **Worries about** | False alarms (FP) | Missed fraud (FN) |
| **Formula** | TP / (TP + FP) | TP / (TP + FN) |
| **Divides by** | Everything I **flagged** | All fraud that **actually existed** |

**Fishing-net analogy:**

- **Recall** = of all the fish in the lake, how many did you pull out? (Did any escape?)
- **Precision** = of everything in your net, how much was actually fish and not weeds? (Is your catch clean?)

A huge net catches every fish (high recall) but scoops up junk too (low precision). A small careful net stays clean (high precision) but lets fish escape (low recall).

**One-line memory hook:**

> **Recall** = "Did I catch all the fraud?" → about what I **missed**
> **Precision** = "Was I right when I cried fraud?" → about my **false alarms**

> There's a **tradeoff**: flag more aggressively → catch more fraud (↑recall) but more false alarms (↓precision), and vice versa.

### F1
A single score that balances precision and recall (their harmonic mean):

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

F1 is only high when *both* precision and recall are decent — a handy one-number summary on imbalanced data.

### ROC-AUC (explained simply)

The model doesn't only say "fraud" or "not fraud" — it gives every transaction a **suspicion score** (how fishy it looks). ROC-AUC asks one simple question:

> If you pick one **real fraud** and one **genuine** transaction at random, how often does the model give the fraud a higher suspicion score than the genuine one?

- **1.0** = it *always* ranks the fraud as more suspicious → perfect.
- **0.5** = it's just guessing (a coin flip) → useless.
- **0.79** = about 79% of the time it correctly rates the fraud as fishier than the genuine one.

The best part: ROC-AUC doesn't depend on where you set your "flag it" cutoff — it just measures how well the model **ranks** fraud above genuine overall. Higher = better at telling them apart.

**Analogy:** imagine a suspicion meter for students. Pick one cheater and one honest student at random — how often does the meter point higher for the cheater? Always → 1.0; random → 0.5.

### Accuracy (and why we avoid it here)

*Accuracy = (TP + TN) / everything.* It looks tempting but is **useless on imbalanced data**: with only ~1.5% fraud, a model that flags nothing scores ~98.5% accuracy while catching zero fraud. That's why this project reports precision, recall, F1, and ROC-AUC instead.
