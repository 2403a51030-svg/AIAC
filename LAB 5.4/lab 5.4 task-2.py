"""
Bias-aware Sentiment Analysis Demo

This script trains a simple sentiment classifier and identifies potential
biases by evaluating performance across sensitive groups (e.g., gender).
It demonstrates a mitigation approach (reweighing) and re-evaluates fairness
metrics post-mitigation.

Notes on bias and responsible ML usage:
- Always examine model metrics by subgroup (e.g., gender, age_group, locale).
- Inspect error types (FPR/FNR) per group; disparate rates can indicate bias.
- Consider data-level mitigations (balancing, augmentation, reweighing),
  algorithm-level mitigations (fairness constraints, adversarial debiasing),
  and post-processing (group thresholds) as appropriate.
- Use domain and legal guidance to select sensitive attributes and targets.
- Avoid storing or logging direct identifiers; only keep anonymized keys.

Dependencies:
    pip install scikit-learn pandas numpy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


# ------------------------------- Data ---------------------------------------

def make_synthetic_dataset(random_seed: int = 42) -> pd.DataFrame:
    """Create a small synthetic dataset for demonstration.

    Columns:
    - text: review text
    - label: 1 positive, 0 negative
    - gender: sensitive attribute for subgroup analysis
    - age_group: additional attribute (not used directly in mitigation)

    The dataset is intentionally imbalanced across groups to surface gaps.
    """
    rng = np.random.default_rng(random_seed)

    positives_male = [
        "I loved this product, it works great!",
        "Fantastic experience, would recommend.",
        "Absolutely brilliant and useful.",
        "Exceeded my expectations.",
    ]
    negatives_male = [
        "Terrible quality, very disappointed.",
        "Did not work as advertised.",
        "Waste of money.",
    ]
    positives_female = [
        "Great value and excellent performance.",
        "Amazing quality, super happy!",
    ]
    negatives_female = [
        "Not good, had many issues.",
        "Poor support and low quality.",
        "I regret this purchase.",
        "Unreliable and frustrating.",
    ]
    positives_nonbinary = [
        "Works fine, met my needs.",
    ]
    negatives_nonbinary = [
        "Mediocre at best.",
        "Underwhelming and buggy.",
    ]

    def rows(texts: List[str], label: int, gender: str, age_group: str) -> List[Dict[str, object]]:
        return [{"text": t, "label": label, "gender": gender, "age_group": age_group} for t in texts]

    data: List[Dict[str, object]] = []
    data += rows(positives_male, 1, "male", "18-34")
    data += rows(negatives_male, 0, "male", "35-54")
    data += rows(positives_female, 1, "female", "18-34")
    data += rows(negatives_female, 0, "female", "35-54")
    data += rows(positives_nonbinary, 1, "nonbinary", "18-34")
    data += rows(negatives_nonbinary, 0, "nonbinary", "35-54")

    df = pd.DataFrame(data)

    # Shuffle rows
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    return df


# -------------------------- Metrics & Fairness -------------------------------

@dataclass
class GroupMetrics:
    count: int
    positive_rate: float
    accuracy: float
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    tpr: Optional[float]  # True positive rate (recall for positive class)
    fpr: Optional[float]  # False positive rate
    fnr: Optional[float]  # False negative rate


def safe_div(numerator: float, denominator: float) -> Optional[float]:
    return float(numerator) / float(denominator) if denominator else None


def compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    group_values: pd.Series,
) -> Dict[str, GroupMetrics]:
    """Compute metrics per group: rates and classification metrics.

    Note: y_score is the predicted probability for positive class. It's not
    used directly in the basic metrics below but is useful for thresholding
    analysis if extended.
    """
    results: Dict[str, GroupMetrics] = {}
    groups = group_values.unique()
    for group in groups:
        mask = group_values == group
        g_y_true = y_true[mask]
        g_y_pred = y_pred[mask]

        count = int(mask.sum())
        positive_rate = float(g_y_pred.mean()) if count else 0.0
        acc = accuracy_score(g_y_true, g_y_pred) if count else 0.0

        # Precision/Recall/F1 can be undefined if a class is absent; handle safely
        try:
            prec = precision_score(g_y_true, g_y_pred, zero_division=0)
            rec = recall_score(g_y_true, g_y_pred, zero_division=0)
            f1 = f1_score(g_y_true, g_y_pred, zero_division=0)
        except Exception:
            prec = rec = f1 = 0.0

        # Confusion elements for rates
        tp = int(((g_y_pred == 1) & (g_y_true == 1)).sum())
        tn = int(((g_y_pred == 0) & (g_y_true == 0)).sum())
        fp = int(((g_y_pred == 1) & (g_y_true == 0)).sum())
        fn = int(((g_y_pred == 0) & (g_y_true == 1)).sum())

        tpr = safe_div(tp, tp + fn)  # equal opportunity metric target
        fpr = safe_div(fp, fp + tn)
        fnr = safe_div(fn, tp + fn)

        results[str(group)] = GroupMetrics(
            count=count,
            positive_rate=positive_rate,
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            tpr=tpr,
            fpr=fpr,
            fnr=fnr,
        )
    return results


def summarize_fairness_gaps(metrics_by_group: Dict[str, GroupMetrics]) -> Dict[str, float]:
    """Compute gap summaries (max-min) across groups for key rates.

    - demographic_parity_gap: gap of positive prediction rates across groups
    - equal_opportunity_gap:  gap of true positive rates (TPR) across groups
    - avg_odds_gap:           average of TPR and FPR gaps
    - fnr_gap:                gap of false negative rates across groups
    """
    def gap(extractor) -> float:
        values = [v for v in (extractor(m) for m in metrics_by_group.values()) if v is not None]
        return (max(values) - min(values)) if values else 0.0

    demographic_parity_gap = gap(lambda m: m.positive_rate)
    tpr_gap = gap(lambda m: m.tpr)
    fpr_gap = gap(lambda m: m.fpr)
    fnr_gap = gap(lambda m: m.fnr)
    avg_odds_gap = (tpr_gap + fpr_gap) / 2.0

    return {
        "demographic_parity_gap": demographic_parity_gap,
        "equal_opportunity_gap": tpr_gap,
        "average_odds_gap": avg_odds_gap,
        "fnr_gap": fnr_gap,
    }


# --------------------------- Mitigation: Reweighing --------------------------

def compute_reweighing_weights(groups: pd.Series, labels: pd.Series) -> np.ndarray:
    """Compute sample weights to balance group-label combinations.

    Kamiran & Calders (2012) "Data preprocessing techniques for classification
    without discrimination": assign weight w(g, y) inversely proportional to
    the frequency of each (group, label) pair so each pair contributes equally.
    We then scale weights to have mean 1 for numerical stability.
    """
    df = pd.DataFrame({"g": groups.astype(str).values, "y": labels.astype(int).values})
    counts = df.value_counts(["g", "y"]).rename("count").reset_index()

    num_groups = df["g"].nunique()
    n = len(df)
    # Target total per (g, y) = n / (num_groups * 2) since binary labels
    target_total = n / float(num_groups * 2)

    # weight(g,y) = target_total / count(g,y)
    weight_lookup: Dict[Tuple[str, int], float] = {}
    for _, row in counts.iterrows():
        key = (str(row["g"]), int(row["y"]))
        weight_lookup[key] = float(target_total) / float(row["count"]) if row["count"] else 0.0

    weights = np.array([weight_lookup[(str(g), int(y))] for g, y in zip(df["g"], df["y"])])

    # Normalize to mean 1
    mean_w = weights.mean() if len(weights) else 1.0
    if mean_w > 0:
        weights = weights / mean_w
    return weights


# ----------------------------- Training & Eval -------------------------------

def train_and_evaluate(df: pd.DataFrame, sensitive_col: str = "gender", random_seed: int = 42) -> None:
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"].astype(int), test_size=0.35, random_state=random_seed, stratify=df["label"].astype(int)
    )
    s_train = df.loc[X_train.index, sensitive_col].astype(str)
    s_test = df.loc[X_test.index, sensitive_col].astype(str)

    # Vectorize
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    # Base model
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    clf.fit(Xtr, y_train)
    proba = clf.predict_proba(Xte)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    # Overall metrics
    print("\n=== Baseline (no mitigation) ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"F1:        {f1_score(y_test, y_pred, zero_division=0):.3f}")

    # Group metrics & fairness
    group_metrics = compute_group_metrics(y_test.values, y_pred, proba, s_test)
    gaps = summarize_fairness_gaps(group_metrics)

    print("\n-- Group metrics (baseline) --")
    for group, m in group_metrics.items():
        print(
            f"{sensitive_col}={group:>9} | n={m.count:2d} | +rate={m.positive_rate:.2f} | "
            f"TPR={m.tpr if m.tpr is not None else float('nan'):.2f} | "
            f"FPR={m.fpr if m.fpr is not None else float('nan'):.2f} | "
            f"FNR={m.fnr if m.fnr is not None else float('nan'):.2f}"
        )

    print("\n-- Fairness gaps (max-min across groups, baseline) --")
    for k, v in gaps.items():
        print(f"{k}: {v:.3f}")

    # ----------------- Mitigation: Reweighing during training -----------------
    weights = compute_reweighing_weights(s_train, y_train)
    clf_rw = LogisticRegression(max_iter=1000, class_weight=None, solver="liblinear")
    clf_rw.fit(Xtr, y_train, sample_weight=weights)
    proba_rw = clf_rw.predict_proba(Xte)[:, 1]
    y_pred_rw = (proba_rw >= 0.5).astype(int)

    print("\n=== After Mitigation (reweighing) ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred_rw):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred_rw, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred_rw, zero_division=0):.3f}")
    print(f"F1:        {f1_score(y_test, y_pred_rw, zero_division=0):.3f}")

    group_metrics_rw = compute_group_metrics(y_test.values, y_pred_rw, proba_rw, s_test)
    gaps_rw = summarize_fairness_gaps(group_metrics_rw)

    print("\n-- Group metrics (reweighing) --")
    for group, m in group_metrics_rw.items():
        print(
            f"{sensitive_col}={group:>9} | n={m.count:2d} | +rate={m.positive_rate:.2f} | "
            f"TPR={m.tpr if m.tpr is not None else float('nan'):.2f} | "
            f"FPR={m.fpr if m.fpr is not None else float('nan'):.2f} | "
            f"FNR={m.fnr if m.fnr is not None else float('nan'):.2f}"
        )

    print("\n-- Fairness gaps (max-min across groups, reweighing) --")
    for k, v in gaps_rw.items():
        print(f"{k}: {v:.3f}")

    print("\nNotes:")
    print("- If gaps shrink post-mitigation, reweighing helped balance subgroup influence.")
    print("- You can further mitigate with data augmentation or group-specific thresholds.")


def main() -> None:
    df = make_synthetic_dataset()
    train_and_evaluate(df, sensitive_col="gender")


if __name__ == "__main__":
    main()


