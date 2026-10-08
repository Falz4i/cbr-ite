import re
import logging
from pathlib import Path
from typing import Iterable, List, Dict

from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as pdf_extract_text
from tqdm import tqdm

from .config import RAW_DOCS_DIR, RAW_TEXT_DIR, LOGS_DIR, PROCESSED_DIR, ensure_dirs

SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".txt"}


def setup_logger() -> logging.Logger:
    ensure_dirs()
    logger = logging.getLogger("cleaning")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(LOGS_DIR / "cleaning.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def iter_source_files(input_dir: Path = RAW_DOCS_DIR) -> List[Path]:
    """Return supported PDF/HTML/TXT files from raw_documents."""
    ensure_dirs()
    files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    return sorted(files)


def extract_text_from_file(path: Path) -> str:
    """Extract text from PDF, HTML, or TXT file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return pdf_extract_text(str(path))
    if suffix in {".html", ".htm"}:
        html = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return soup.get_text(" ")
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {path}")


def clean_text(text: str) -> str:
    """Clean Indonesian court-decision text while keeping legal information."""
    if not text:
        return ""

    replacements = {
        "\ufeff": " ",
        "\u00a0": " ",
        "\t": " ",
        "\r": "\n",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Common noise from scanned/generated decision files.
    noise_patterns = [
        r"Direktori Putusan Mahkamah Agung Republik Indonesia",
        r"putusan\.mahkamahagung\.go\.id",
        r"Halaman\s+\d+\s+dari\s+\d+",
        r"Disclaimer.*?(?=P U T U S A N|PUTUSAN|Nomor|$)",
        r"Mahkamah Agung Republik Indonesia",
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE | re.DOTALL)

    # Normalize unusual spacing in headings, but do not remove legal punctuation.
    text = re.sub(r"P\s*U\s*T\s*U\s*S\s*A\s*N", "PUTUSAN", text, flags=re.IGNORECASE)
    text = re.sub(r"M\s*E\s*N\s*G\s*A\s*D\s*I\s*L\s*I", "MENGADILI", text, flags=re.IGNORECASE)
    text = re.sub(r"M\s*E\s*N\s*I\s*M\s*B\s*A\s*N\s*G", "MENIMBANG", text, flags=re.IGNORECASE)

    # Normalize whitespace.
    text = re.sub(r"[ \f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def process_raw_documents(input_dir: Path = RAW_DOCS_DIR, output_dir: Path = RAW_TEXT_DIR) -> List[Dict[str, str]]:
    """Extract and clean all source files into data/raw/case_XXX.txt."""
    ensure_dirs()
    logger = setup_logger()
    output_dir.mkdir(parents=True, exist_ok=True)
    files = iter_source_files(input_dir)

    if not files:
        logger.warning("No source files found in %s", input_dir)
        return []

    mapping = []
    for idx, src in enumerate(tqdm(files, desc="Preprocessing putusan"), start=1):
        case_id = f"case_{idx:03d}"
        out_path = output_dir / f"{case_id}.txt"
        try:
            raw_text = extract_text_from_file(src)
            cleaned = clean_text(raw_text)
            out_path.write_text(cleaned, encoding="utf-8")
            word_count = len(cleaned.split())
            status = "ok" if word_count >= 100 else "warning_short_text"
            logger.info("%s -> %s | words=%s | status=%s", src.name, out_path.name, word_count, status)
            mapping.append(
                {
                    "case_id": case_id,
                    "source_file": str(src.relative_to(input_dir)),
                    "raw_text_file": str(out_path.relative_to(output_dir.parent)),
                    "word_count": word_count,
                    "status": status,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed processing %s: %s", src, exc)
            mapping.append(
                {
                    "case_id": case_id,
                    "source_file": str(src.relative_to(input_dir)),
                    "raw_text_file": "",
                    "word_count": 0,
                    "status": f"failed: {exc}",
                }
            )

    try:
        import pandas as pd

        pd.DataFrame(mapping).to_csv(PROCESSED_DIR / "source_files.csv", index=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save source_files.csv: %s", exc)

    return mapping


if __name__ == "__main__":
    process_raw_documents()
