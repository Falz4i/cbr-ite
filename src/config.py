from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_documents"
RAW_TEXT_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"
RESULTS_DIR = DATA_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"

CASES_CSV = PROCESSED_DIR / "cases.csv"
CASES_JSON = PROCESSED_DIR / "cases.json"
QUERIES_JSON = EVAL_DIR / "queries.json"
RETRIEVAL_RESULTS_CSV = RESULTS_DIR / "retrieval_results.csv"
PREDICTIONS_CSV = RESULTS_DIR / "predictions.csv"
RETRIEVAL_METRICS_CSV = EVAL_DIR / "retrieval_metrics.csv"
PREDICTION_METRICS_CSV = EVAL_DIR / "prediction_metrics.csv"

TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
TFIDF_MATRIX_PATH = MODELS_DIR / "tfidf_matrix.joblib"
CASE_INDEX_PATH = MODELS_DIR / "case_index.joblib"


def ensure_dirs() -> None:
    """Create required folders if they do not exist."""
    for path in [
        DATA_DIR,
        RAW_DOCS_DIR,
        RAW_TEXT_DIR,
        PROCESSED_DIR,
        EVAL_DIR,
        RESULTS_DIR,
        LOGS_DIR,
        MODELS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
