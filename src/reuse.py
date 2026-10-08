from collections import defaultdict
from typing import Dict, Any

import pandas as pd

from .config import PREDICTIONS_CSV, RESULTS_DIR, ensure_dirs
from .retrieval import retrieve, predict_label_with_classifier


def predict_outcome(query: str, k: int = 5) -> Dict[str, Any]:
    """Reuse the solution from top-k retrieved cases and combine it with a simple label classifier."""
    ensure_dirs()
    top_k = retrieve(query, k=k, save=True)
    if top_k.empty:
        return {
            "query": query,
            "predicted_label": "tidak diketahui",
            "predicted_solution": "Tidak ada kasus pembanding yang ditemukan.",
            "top_5_case_ids": "",
            "reuse_strategy": "no_retrieval_result",
        }

    label_scores = defaultdict(float)
    for _, row in top_k.iterrows():
        label = row.get("label_putusan", "tidak diketahui") or "tidak diketahui"
        label_scores[str(label)] += float(row.get("similarity", 0.0))

    predicted_label_retrieval = max(label_scores.items(), key=lambda item: item[1])[0]
    classifier_result = predict_label_with_classifier(query)
    predicted_label_classifier = classifier_result.get("predicted_label_classifier", predicted_label_retrieval)

    best_case = top_k.iloc[0]
    predicted_solution = best_case.get("amar_putusan", "")
    if not isinstance(predicted_solution, str) or not predicted_solution.strip():
        predicted_solution = f"Solusi mengikuti pola kasus terdekat: {best_case.get('no_perkara', best_case.get('case_id'))}"

    result = {
        "query": query,
        "predicted_label": predicted_label_retrieval,
        "predicted_label_retrieval": predicted_label_retrieval,
        "predicted_label_classifier": predicted_label_classifier,
        "predicted_solution": predicted_solution,
        "reuse_strategy": "weighted_vote_top_k_plus_top_1_amar",
        "top_5_case_ids": ",".join(top_k["case_id"].astype(str).tolist()),
        "top_1_case_id": str(best_case.get("case_id", "")),
        "top_1_no_perkara": str(best_case.get("no_perkara", "")),
        "top_1_similarity": float(best_case.get("similarity", 0.0)),
        "label_probabilities": classifier_result.get("label_probabilities", {}),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(PREDICTIONS_CSV, index=False)
    return result


if __name__ == "__main__":
    sample_query = "terdakwa menyebarkan informasi elektronik bermuatan penghinaan melalui media sosial"
    print(predict_outcome(sample_query, k=5))
