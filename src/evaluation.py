import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .config import (
    QUERIES_JSON,
    RETRIEVAL_METRICS_CSV,
    PREDICTION_METRICS_CSV,
    EVAL_DIR,
    ensure_dirs,
)
from .retrieval import retrieve
from .reuse import predict_outcome


def load_queries(path: Path = QUERIES_JSON) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def evaluate_retrieval(k: int = 5, queries_path: Path = QUERIES_JSON) -> pd.DataFrame:
    """Evaluate retrieval using hit@k, top-1 accuracy, precision@k, recall@k, and F1@k."""
    ensure_dirs()
    queries = load_queries(queries_path)
    rows = []
    for item in queries:
        query_id = item.get("query_id", "")
        query = item.get("query", "")
        ground_truth = item.get("ground_truth_case_id", "")
        if not query or not ground_truth:
            continue
        results = retrieve(query, k=k, save=False)
        retrieved_ids = results["case_id"].astype(str).tolist() if not results.empty else []
        top_1 = retrieved_ids[0] if retrieved_ids else ""
        precision_at_k = (1 / k) if str(ground_truth) in retrieved_ids else 0.0
        recall_at_k = 1.0 if str(ground_truth) in retrieved_ids else 0.0
        f1_at_k = 0.0 if precision_at_k + recall_at_k == 0 else (2 * precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
        rows.append(
            {
                "query_id": query_id,
                "query": query,
                "ground_truth_case_id": ground_truth,
                "top_1_case_id": top_1,
                "top_k_case_ids": ",".join(retrieved_ids),
                "hit_at_k": int(str(ground_truth) in retrieved_ids),
                "top_1_correct": int(str(ground_truth) == top_1),
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k,
                "f1_at_k": f1_at_k,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        summary = pd.DataFrame(
            [
                {
                    "query_id": "SUMMARY",
                    "query": "",
                    "ground_truth_case_id": "",
                    "top_1_case_id": "",
                    "top_k_case_ids": "",
                    "hit_at_k": df["hit_at_k"].mean(),
                    "top_1_correct": df["top_1_correct"].mean(),
                    "precision_at_k": df["precision_at_k"].mean(),
                    "recall_at_k": df["recall_at_k"].mean(),
                    "f1_at_k": df["f1_at_k"].mean(),
                }
            ]
        )
        df = pd.concat([df, summary], ignore_index=True)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RETRIEVAL_METRICS_CSV, index=False)
    return df


def evaluate_prediction(k: int = 5, queries_path: Path = QUERIES_JSON) -> pd.DataFrame:
    """Evaluate predicted outcome labels when expected_label exists in queries.json."""
    ensure_dirs()
    queries = load_queries(queries_path)
    rows = []
    y_true, y_pred = [], []

    for item in queries:
        query = item.get("query", "")
        expected = item.get("expected_label", "")
        if not query or not expected:
            continue
        pred = predict_outcome(query, k=k)
        predicted_label = pred.get("predicted_label", "tidak diketahui")
        y_true.append(str(expected).lower())
        y_pred.append(str(predicted_label).lower())
        rows.append(
            {
                "query_id": item.get("query_id", ""),
                "query": query,
                "expected_label": expected,
                "predicted_label": predicted_label,
                "predicted_label_classifier": pred.get("predicted_label_classifier", ""),
                "top_1_case_id": pred.get("top_1_case_id", ""),
                "top_1_similarity": pred.get("top_1_similarity", 0.0),
                "correct": int(str(expected).lower() == str(predicted_label).lower()),
            }
        )

    df = pd.DataFrame(rows)
    if y_true:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
        summary = pd.DataFrame(
            [
                {
                    "query_id": "SUMMARY",
                    "query": "",
                    "expected_label": "",
                    "predicted_label": "",
                    "predicted_label_classifier": "",
                    "top_1_case_id": "",
                    "top_1_similarity": "",
                    "correct": accuracy_score(y_true, y_pred),
                    "precision_macro": precision,
                    "recall_macro": recall,
                    "f1_macro": f1,
                }
            ]
        )
        df = pd.concat([df, summary], ignore_index=True)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PREDICTION_METRICS_CSV, index=False)
    return df


def build_failure_analysis(retrieval_df: pd.DataFrame, prediction_df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact failure-analysis table for notebook discussion."""
    rows = []
    detail = retrieval_df[retrieval_df.get("query_id", "") != "SUMMARY"].copy() if not retrieval_df.empty else pd.DataFrame()
    for _, row in detail.iterrows():
        failure_type = []
        if int(row.get("hit_at_k", 0)) == 0:
            failure_type.append("ground truth tidak masuk top-k")
        if int(row.get("top_1_correct", 0)) == 0:
            failure_type.append("top-1 berbeda dari ground truth")
        if not failure_type:
            failure_type.append("retrieval sesuai")
        rows.append(
            {
                "query_id": row.get("query_id", ""),
                "failure_type": "; ".join(failure_type),
                "possible_cause": "query terlalu pendek, variasi istilah hukum, metadata belum lengkap, atau ground truth perlu diverifikasi manual",
                "recommendation": "tambah query uji berlabel, bersihkan OCR/PDF lebih detail, gunakan field pasal dan amar sebagai bobot tambahan, lalu validasi ground truth oleh pembaca hukum",
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(evaluate_retrieval(k=5))
    print(evaluate_prediction(k=5))
