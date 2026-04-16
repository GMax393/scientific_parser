import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Tuple
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer


ANNOTATED_DATA_PATH = "data/annotated_dataset.json"
MODEL_PATH = "models/block_classifier.joblib"
MODEL_META_PATH = "models/block_classifier_meta.json"
REPORTS_DIR = "reports"
METRICS_PATH = os.path.join(REPORTS_DIR, "metrics.json")
HARD_EXAMPLES_PATH = os.path.join(REPORTS_DIR, "hard_examples.csv")


LABEL_MAP = {
    "author_candidate": "author",
    "journal_info": "journal",
    "year_candidate": "year",
    "doi_candidate": "doi",
    "content": "other",
    "abstract": "other",
}


def _normalize_domain(url: str) -> str:
    host = (urlparse(url or "").netloc or "").lower()
    return host.replace("www.", "") if host else "unknown"


def load_annotated_dataset(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_blocks(pages: List[Dict]) -> pd.DataFrame:
    rows = []
    for page in pages:
        url = page.get("url", "")
        source_type = page.get("source_type", "")
        domain = _normalize_domain(url)
        for block in page.get("text_blocks", []):
            text = (block.get("text") or "").strip()
            if len(text) < 5:
                continue
            digits = sum(ch.isdigit() for ch in text)
            alpha = sum(ch.isalpha() for ch in text)
            uppers = sum(ch.isupper() for ch in text if ch.isalpha())
            rows.append(
                {
                    "text": text,
                    "text_char": text,
                    "tag": block.get("tag", ""),
                    "classes": " ".join(block.get("classes", [])),
                    "text_length": int(block.get("text_length", len(text))),
                    "word_count": int(block.get("word_count", len(text.split()))),
                    "is_visible": int(block.get("is_visible", True)),
                    "dom_depth": float(block.get("dom_depth", 0)),
                    "sibling_index": float(block.get("sibling_index", 0)),
                    "link_density": float(block.get("link_density", 0.0)),
                    "digit_ratio": float(digits / max(len(text), 1)),
                    "upper_ratio": float(uppers / max(alpha, 1)),
                    "page_url": url,
                    "domain": domain,
                    "source_type": source_type,
                    "label": LABEL_MAP.get(block.get("label", "other"), block.get("label", "other")),
                }
            )
    return pd.DataFrame(rows)


def clean_and_rebalance(df: pd.DataFrame, other_ratio: float = 1.8) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["page_url", "text", "tag"]).copy()
    informative = df[df["label"] != "other"]
    other = df[df["label"] == "other"]
    if not informative.empty and not other.empty:
        max_other = int(len(informative) * other_ratio)
        if len(other) > max_other > 0:
            other = other.sample(n=max_other, random_state=42)
            df = pd.concat([informative, other], ignore_index=True)
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)


def build_preprocessor(text_max_features: int = 12000) -> ColumnTransformer:
    numeric_features = [
        "text_length",
        "word_count",
        "is_visible",
        "dom_depth",
        "sibling_index",
        "link_density",
        "digit_ratio",
        "upper_ratio",
    ]
    categorical_features = ["tag", "classes", "source_type", "domain"]
    return ColumnTransformer(
        transformers=[
            ("text_word", TfidfVectorizer(max_features=text_max_features, ngram_range=(1, 2), min_df=2), "text"),
            ("text_char", TfidfVectorizer(max_features=5000, analyzer="char_wb", ngram_range=(3, 5), min_df=2), "text_char"),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )


def build_model_candidates() -> Dict[str, object]:
    return {
        "logreg_balanced": LogisticRegression(
            max_iter=8000,
            tol=1e-3,
            solver="saga",
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
            min_samples_leaf=1,
        ),
        "linear_svc_calibrated": CalibratedClassifierCV(
            estimator=LinearSVC(
                class_weight="balanced",
                random_state=42,
                dual="auto",
                max_iter=12000,
                tol=1e-3,
            ),
            method="sigmoid",
            cv=3,
        ),
    }


def split_train_test(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    groups = X["domain"].fillna("unknown")
    unique_groups = groups.nunique()
    if unique_groups >= 5:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx], "group_by_domain"
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, "stratified_random"


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def evaluate_by_domain(X_test: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    domain_metrics: Dict[str, Dict[str, float]] = {}
    eval_df = X_test.copy()
    eval_df["y_true"] = y_true.values
    eval_df["y_pred"] = y_pred
    for domain, part in eval_df.groupby("domain"):
        if len(part) < 3:
            continue
        domain_metrics[domain] = {
            "samples": int(len(part)),
            "accuracy": float(accuracy_score(part["y_true"], part["y_pred"])),
            "f1_macro": float(f1_score(part["y_true"], part["y_pred"], average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(part["y_true"], part["y_pred"], average="weighted", zero_division=0)),
        }
    return domain_metrics


def collect_hard_examples(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    out_path: str,
    limit: int = 200,
) -> None:
    probs = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_test)
        except Exception:
            probs = None

    rows = []
    for i, (truth, pred) in enumerate(zip(y_test.tolist(), y_pred.tolist())):
        confidence = None
        if probs is not None:
            confidence = float(np.max(probs[i]))
        if truth != pred or (confidence is not None and confidence < 0.55):
            x = X_test.iloc[i]
            rows.append(
                {
                    "domain": x.get("domain", ""),
                    "url": x.get("page_url", ""),
                    "tag": x.get("tag", ""),
                    "classes": x.get("classes", ""),
                    "label_true": truth,
                    "label_pred": pred,
                    "confidence": confidence,
                    "text": x.get("text", "")[:500],
                }
            )
    if not rows:
        return
    pd.DataFrame(rows).head(limit).to_csv(out_path, index=False, encoding="utf-8")


def run_training_pipeline() -> Dict:
    if not os.path.exists(ANNOTATED_DATA_PATH):
        raise FileNotFoundError(f"Файл разметки не найден: {ANNOTATED_DATA_PATH}")

    pages = load_annotated_dataset(ANNOTATED_DATA_PATH)
    df = flatten_blocks(pages)
    df = clean_and_rebalance(df)
    if df.empty:
        raise RuntimeError("После очистки датасет пуст.")

    print(f"Всего блоков для обучения: {len(df)}")
    print("Распределение классов:")
    print(df["label"].value_counts())

    X = df.drop(columns=["label"])
    y = df["label"]
    X_train, X_test, y_train, y_test, split_strategy = split_train_test(X, y)

    preprocessor = build_preprocessor()
    model_candidates = build_model_candidates()

    model_reports: Dict[str, Dict] = {}
    best_name = None
    best_score = -1.0
    best_pipeline = None
    best_pred = None

    for model_name, clf in model_candidates.items():
        print(f"\n=== Обучение: {model_name} ===")
        pipeline = Pipeline([("preprocessor", preprocessor), ("clf", clf)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        summary = evaluate_predictions(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        domain_report = evaluate_by_domain(X_test, y_test, y_pred)

        model_reports[model_name] = {
            "summary": summary,
            "classification_report": report_dict,
            "domain_metrics": domain_report,
        }

        print(f"accuracy={summary['accuracy']:.4f} f1_macro={summary['f1_macro']:.4f} f1_weighted={summary['f1_weighted']:.4f}")

        score = summary["f1_macro"] * 0.7 + summary["f1_weighted"] * 0.3
        if score > best_score:
            best_score = score
            best_name = model_name
            best_pipeline = pipeline
            best_pred = y_pred

    if best_pipeline is None or best_name is None or best_pred is None:
        raise RuntimeError("Не удалось выбрать лучшую модель.")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    joblib.dump(best_pipeline, MODEL_PATH)

    model_meta = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_model": best_name,
        "split_strategy": split_strategy,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "labels": sorted(y.unique().tolist()),
        "feature_notes": {
            "text": "word + char tfidf",
            "categorical": ["tag", "classes", "source_type", "domain"],
            "numeric": [
                "text_length",
                "word_count",
                "is_visible",
                "dom_depth",
                "sibling_index",
                "link_density",
                "digit_ratio",
                "upper_ratio",
            ],
        },
    }
    with open(MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(model_meta, f, indent=2, ensure_ascii=False)

    collect_hard_examples(best_pipeline, X_test, y_test, best_pred, HARD_EXAMPLES_PATH)

    metrics_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data": {
            "total_samples": int(len(df)),
            "class_distribution": dict(Counter(y.tolist())),
            "domain_distribution_top10": dict(Counter(X["domain"].tolist()).most_common(10)),
            "split_strategy": split_strategy,
        },
        "models": model_reports,
        "best_model": best_name,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    print(f"\nЛучшая модель: {best_name}")
    print(f"Сохранено: {MODEL_PATH}")
    print(f"Метрики: {METRICS_PATH}")
    print(f"Сложные примеры: {HARD_EXAMPLES_PATH}")
    return metrics_payload


if __name__ == "__main__":
    run_training_pipeline()
