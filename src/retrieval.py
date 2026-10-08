from pathlib import Path
from typing import List, Dict, Any

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .config import (
    CASES_CSV,
    MODELS_DIR,
    RESULTS_DIR,
    RETRIEVAL_RESULTS_CSV,
    TFIDF_VECTORIZER_PATH,
    TFIDF_MATRIX_PATH,
    CASE_INDEX_PATH,
    ensure_dirs,
)

LABEL_CLASSIFIER_PATH = MODELS_DIR / "label_classifier.joblib"

INDONESIAN_STOPWORDS = [
    "yang", "dan", "di", "ke", "dari", "dengan", "untuk", "pada", "dalam", "adalah",
    "atau", "ini", "itu", "sebagai", "oleh", "karena", "bahwa", "telah", "para", "akan",
    "tidak", "tersebut", "serta", "maka", "dapat", "atas", "juga", "putusan", "pengadilan",
    "tinggi", "negeri", "surabaya", "terdakwa", "hakim", "penuntut", "umum", "perkara",
]


def _make_retrieval_text(row: pd.Series) -> str:
    value = str(row.get("retrieval_text", ""))
    if value and value.lower() != "nan":
        return value
    parts = [
        str(row.get("jenis_perkara", "")),
        str(row.get("ringkasan_fakta", "")),
        str(row.get("argumen_hukum", "")),
        str(row.get("pasal", "")),
        str(row.get("amar_putusan", "")),
        str(row.get("text_full", ""))[:20000],
    ]
    return " ".join(parts)


def _new_vectorizer(max_features: int = 30000) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,
        stop_words=INDONESIAN_STOPWORDS,
        sublinear_tf=True,
        norm="l2",
    )


def fit_retrieval_model(cases_csv: Path = CASES_CSV) -> Dict[str, object]:
    """Fit TF-IDF retrieval model and a simple Naive Bayes label classifier."""
    ensure_dirs()
    if not cases_csv.exists():
        raise FileNotFoundError(f"Cases file not found: {cases_csv}. Run representation stage first.")

    df = pd.read_csv(cases_csv).fillna("")
    if df.empty:
        raise ValueError("cases.csv is empty. Add putusan files and run preprocess + representation stages.")

    retrieval_texts = df.apply(_make_retrieval_text, axis=1).tolist()
    vectorizer = _new_vectorizer(max_features=30000)
    matrix = vectorizer.fit_transform(retrieval_texts)

    case_columns = [
        "case_id", "no_perkara", "tanggal_putusan", "pengadilan", "jenis_perkara",
        "pasal", "pihak", "label_putusan", "amar_putusan", "source_file",
    ]
    case_index = df[[c for c in case_columns if c in df.columns]].to_dict("records")

    labels = df.get("label_putusan", pd.Series(["tidak diketahui"] * len(df))).astype(str).replace("", "tidak diketahui")
    if labels.nunique() > 1:
        classifier = Pipeline([
            ("tfidf", _new_vectorizer(max_features=20000)),
            ("clf", MultinomialNB(alpha=0.5)),
        ])
    else:
        classifier = DummyClassifier(strategy="most_frequent")
    classifier.fit(retrieval_texts, labels)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)
    joblib.dump(matrix, TFIDF_MATRIX_PATH)
    joblib.dump(case_index, CASE_INDEX_PATH)
    joblib.dump(classifier, LABEL_CLASSIFIER_PATH)

    return {
        "num_cases": int(len(df)),
        "num_features": int(len(vectorizer.get_feature_names_out())),
        "num_labels": int(labels.nunique()),
        "label_distribution": labels.value_counts().to_dict(),
    }


def load_retrieval_artifacts():
    if not (TFIDF_VECTORIZER_PATH.exists() and TFIDF_MATRIX_PATH.exists() and CASE_INDEX_PATH.exists()):
        fit_retrieval_model()
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    matrix = joblib.load(TFIDF_MATRIX_PATH)
    case_index = joblib.load(CASE_INDEX_PATH)
    return vectorizer, matrix, case_index


def load_label_classifier():
    if not LABEL_CLASSIFIER_PATH.exists():
        fit_retrieval_model()
    return joblib.load(LABEL_CLASSIFIER_PATH)


def retrieve(query: str, k: int = 5, save: bool = True) -> pd.DataFrame:
    """Retrieve top-k most similar cases for a new query using cosine similarity."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    vectorizer, matrix, case_index = load_retrieval_artifacts()
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    top_indices = scores.argsort()[::-1][:k]

    rows: List[Dict[str, Any]] = []
    for rank, idx in enumerate(top_indices, start=1):
        meta = case_index[int(idx)].copy()
        meta.update({"rank": rank, "similarity": round(float(scores[int(idx)]), 6), "query": query})
        rows.append(meta)

    results = pd.DataFrame(rows)
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results.to_csv(RETRIEVAL_RESULTS_CSV, index=False)
    return results


def predict_label_with_classifier(query: str) -> Dict[str, Any]:
    classifier = load_label_classifier()
    label = classifier.predict([query])[0]
    result: Dict[str, Any] = {"predicted_label_classifier": str(label)}
    if hasattr(classifier, "predict_proba"):
        try:
            proba = classifier.predict_proba([query])[0]
            classes = list(classifier.classes_)
            result["label_probabilities"] = {str(c): round(float(p), 6) for c, p in zip(classes, proba)}
        except Exception:
            result["label_probabilities"] = {}
    return result


if __name__ == "__main__":
    info = fit_retrieval_model()
    print(info)
