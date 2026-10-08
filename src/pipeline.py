import argparse
from pprint import pprint

from .config import ensure_dirs, RAW_DOCS_DIR, RAW_TEXT_DIR, CASES_CSV
from .preprocess import process_raw_documents
from .extract_metadata import build_case_representation
from .retrieval import fit_retrieval_model, retrieve
from .reuse import predict_outcome
from .evaluation import evaluate_retrieval, evaluate_prediction


def run_stage(stage: str, query: str = "", k: int = 5):
    ensure_dirs()

    if stage == "preprocess":
        mapping = process_raw_documents()
        print(f"Preprocessing selesai. Jumlah file diproses: {len(mapping)}")
        print(f"Input : {RAW_DOCS_DIR}")
        print(f"Output: {RAW_TEXT_DIR}")
        return mapping

    if stage == "represent":
        df = build_case_representation()
        print(f"Case representation selesai. Jumlah case: {len(df)}")
        print(f"Output: {CASES_CSV}")
        return df

    if stage == "retrieval":
        info = fit_retrieval_model()
        print("Model retrieval selesai dibuat:")
        pprint(info)
        if query:
            print(retrieve(query, k=k))
        return info

    if stage == "predict":
        if not query:
            raise ValueError("Stage predict membutuhkan --query")
        result = predict_outcome(query, k=k)
        pprint(result)
        return result

    if stage == "evaluate":
        retrieval_metrics = evaluate_retrieval(k=k)
        prediction_metrics = evaluate_prediction(k=k)
        print("Retrieval metrics:")
        print(retrieval_metrics)
        print("Prediction metrics:")
        print(prediction_metrics)
        return retrieval_metrics, prediction_metrics

    raise ValueError(f"Unknown stage: {stage}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline CBR Putusan ITE PN Surabaya")
    parser.add_argument("--stage", choices=["preprocess", "represent", "retrieval", "predict", "evaluate"], help="Tahap yang ingin dijalankan")
    parser.add_argument("--all", action="store_true", help="Jalankan semua tahap utama")
    parser.add_argument("--query", default="", help="Query kasus baru untuk retrieval/prediksi")
    parser.add_argument("--k", type=int, default=5, help="Jumlah top-k kasus termirip")
    args = parser.parse_args()

    ensure_dirs()

    if args.all:
        run_stage("preprocess", k=args.k)
        run_stage("represent", k=args.k)
        run_stage("retrieval", query=args.query, k=args.k)
        if args.query:
            run_stage("predict", query=args.query, k=args.k)
        run_stage("evaluate", k=args.k)
        return

    if args.stage:
        run_stage(args.stage, query=args.query, k=args.k)
        return

    if args.query:
        print(retrieve(args.query, k=args.k))
        pprint(predict_outcome(args.query, k=args.k))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
