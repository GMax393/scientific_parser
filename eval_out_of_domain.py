"""
Оценка классификатора блоков на доменах, полностью исключённых из обучения (true OOD).

Обучает модель только на «видимых» доменах и считает метрики на удерживаемых доменах.
Запуск из каталога проекта:  python eval_out_of_domain.py
  python eval_out_of_domain.py --holdout "example.org,other.net"
  python eval_out_of_domain.py --auto-n 3
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import List

from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

from train_evaluate import (
    ANNOTATED_DATA_PATH,
    build_model_candidates,
    build_preprocessor,
    clean_and_rebalance,
    flatten_blocks,
    load_annotated_dataset,
)


def _pick_smallest_domains(df, n: int, min_rows: int = 15) -> List[str]:
    sizes = df.groupby("domain").size().sort_values()
    picked: List[str] = []
    for dom, sz in sizes.items():
        if dom == "unknown" or sz < min_rows:
            continue
        picked.append(dom)
        if len(picked) >= n:
            break
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description="OOD evaluation by held-out domains")
    parser.add_argument(
        "--holdout",
        type=str,
        default="",
        help="Список доменов через запятую (как в колонке domain после нормализации)",
    )
    parser.add_argument("--auto-n", type=int, default=2, help="Автовыбор N наименьших подходящих доменов")
    parser.add_argument(
        "--model",
        type=str,
        default="logreg_balanced",
        choices=list(build_model_candidates().keys()),
        help="Какой классификатор обучить на train-доменах (для скорости по умолчанию logreg)",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Опционально: путь к JSON-файлу (если не задан — только вывод в консоль)",
    )
    args = parser.parse_args()

    if not os.path.exists(ANNOTATED_DATA_PATH):
        raise SystemExit(f"Нет данных: {ANNOTATED_DATA_PATH}")

    pages = load_annotated_dataset(ANNOTATED_DATA_PATH)
    df = clean_and_rebalance(flatten_blocks(pages))
    if df.empty:
        raise SystemExit("Датасет пуст")

    df["domain"] = df["domain"].fillna("unknown").astype(str).str.lower()

    if args.holdout.strip():
        holdout = [x.strip().lower() for x in args.holdout.split(",") if x.strip()]
    else:
        holdout = _pick_smallest_domains(df, n=max(1, args.auto_n))

    if not holdout:
        raise SystemExit("Не удалось выбрать домены для удержания. Задайте --holdout вручную.")

    holdout_set = set(holdout)
    missing = holdout_set - set(df["domain"].unique())
    if missing:
        raise SystemExit(f"Домены отсутствуют в данных: {sorted(missing)}")

    train_df = df[~df["domain"].isin(holdout_set)].copy()
    test_df = df[df["domain"].isin(holdout_set)].copy()
    if train_df.empty or test_df.empty:
        raise SystemExit("После разбиения train или test пуст — измените список доменов.")

    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    clf = build_model_candidates()[args.model]
    pipeline = Pipeline([("preprocessor", build_preprocessor()), ("clf", clf)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    summary = {
        "accuracy": float((y_test.values == y_pred).mean()),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    per_domain = {}
    tmp = test_df.copy()
    tmp["y_true"] = y_test.values
    tmp["y_pred"] = y_pred
    for dom, part in tmp.groupby("domain"):
        if len(part) < 2:
            continue
        per_domain[dom] = {
            "samples": int(len(part)),
            "accuracy": float((part["y_true"].values == part["y_pred"].values).mean()),
            "f1_macro": float(f1_score(part["y_true"], part["y_pred"], average="macro", zero_division=0)),
        }

    out = {
        "holdout_domains": sorted(holdout_set),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "model_trained": args.model,
        "summary": summary,
        "classification_report": report,
        "per_domain_in_holdout": per_domain,
        "class_distribution_test": dict(Counter(y_test.tolist())),
    }

    if args.save_json.strip():
        path = args.save_json.strip()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"JSON записан: {path}")

    print("Домены вне обучения:", ", ".join(sorted(holdout_set)))
    print(f"Train строк: {len(train_df)}, test строк: {len(test_df)}")
    print(f"accuracy={summary['accuracy']:.4f} f1_macro={summary['f1_macro']:.4f} f1_weighted={summary['f1_weighted']:.4f}")


if __name__ == "__main__":
    main()
