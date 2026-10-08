import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import RAW_TEXT_DIR, CASES_CSV, CASES_JSON, PROCESSED_DIR, ensure_dirs

MONTHS_ID = (
    "januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember|nopember|pebruari"
)


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip(" ;,.:-")


def _search(pattern: str, text: str, flags: int = re.IGNORECASE | re.DOTALL) -> str:
    match = re.search(pattern, text, flags)
    if not match:
        return ""
    for group in match.groups():
        if group:
            return _compact(group)
    return _compact(match.group(0))


def _clean_case_number(value: str) -> str:
    value = _compact(value).upper()
    value = re.sub(r"\s*/\s*", "/", value)
    value = re.sub(r"PID\s*\.\s*SUS", "PID.SUS", value)
    value = re.sub(r"/(PT|PN)(?=[A-Z])", r"/\1 ", value)
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\b(PN|PT)\s+([A-Z.]+)\b", r"\1 \2", value)
    # Some PDFs concatenate the court code with the next heading, e.g. PT SBYDEMI.
    for code in ["SBY", "SDA", "BYW", "MLG", "MJK", "PMK"]:
        value = re.sub(rf"\b(PT|PN)\s+{code}[A-Z.]*\b", rf"\1 {code}", value)
    return value.strip(" .;,")


def extract_no_perkara(text: str) -> str:
    """Extract the main appeal/court case number without swallowing the header."""
    patterns = [
        r"\b\d{1,6}\s*/\s*PID\s*\.?\s*SUS\s*/\s*\d{4}\s*/\s*(?:PT|PN)\s*[A-Z.]{2,12}\b",
        r"Nomor\s*[:\-]?\s*(\d{1,6}\s*/\s*Pid\s*\.?\s*Sus\s*/\s*\d{4}\s*/\s*(?:PT|PN)\s*[A-Za-z.]{2,12})",
        r"Perkara\s+No\s*[:\-]?\s*(\d{1,6}\s*/\s*Pid\s*\.?\s*Sus\s*/\s*\d{4}\s*/\s*(?:PT|PN)\s*[A-Za-z.]{2,12})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = match.group(1) if match.groups() else match.group(0)
            return _clean_case_number(value)
    return ""


def extract_tanggal_putusan(text: str) -> str:
    """Prefer the final judgment date near 'Demikian diputus'."""
    date_pattern = rf"\d{{1,2}}\s+(?:{MONTHS_ID})\s+\d{{4}}"
    candidates: List[str] = []

    for pattern in [
        rf"Demikian\s+diputus.*?tanggal\s+({date_pattern})",
        rf"putusan\s+tersebut\s+diucapkan.*?tanggal\s+({date_pattern})",
        rf"diputuskan\s+dalam.*?tanggal\s+({date_pattern})",
        rf"pada\s+hari\s+.*?tanggal\s+({date_pattern})",
        rf"tanggal\s+putusan\s*[:\-]?\s*({date_pattern})",
    ]:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            candidates.append(_compact(match.group(1)))
    if candidates:
        return candidates[-1]

    all_dates = re.findall(date_pattern, text, flags=re.IGNORECASE)
    return _compact(all_dates[-1]) if all_dates else ""


def extract_pengadilan(text: str) -> str:
    if re.search(r"Pengadilan\s+Tinggi\s+Surabaya|PT\s*SBY", text, re.IGNORECASE):
        return "Pengadilan Tinggi Surabaya"
    value = _search(r"Pengadilan\s+Negeri\s+([A-Za-z ]{3,40})", text)
    if value:
        return "Pengadilan Negeri " + value.title()
    if re.search(r"PN\s*Sby|PN\s*Surabaya|Pengadilan Negeri Surabaya", text, re.IGNORECASE):
        return "Pengadilan Negeri Surabaya"
    return "Tidak teridentifikasi"


def infer_jenis_perkara(text: str) -> str:
    low = text.lower()
    if "informasi elektronik" in low or "transaksi elektronik" in low or "uu ite" in low:
        return "Pidana Khusus ITE"
    if "narkotika" in low or "psikotropika" in low:
        return "Pidana Khusus Narkotika/Psikotropika"
    if "pid.sus" in low:
        return "Pidana Khusus"
    return "Pidana"


def extract_pasal(text: str) -> str:
    matches = re.findall(
        r"Pasal\s+\d+[A-Za-z]?(?:\s+ayat\s*\(?\d+\)?)?(?:\s+jo\.?\s+Pasal\s+\d+[A-Za-z]?(?:\s+ayat\s*\(?\d+\)?)?)?(?:\s+(?:Undang-Undang|UU|KUHP|KUHAP)[^.;,]{0,90})?",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = []
    seen = set()
    for m in matches:
        m = _compact(m)
        m = re.sub(r"\s+", " ", m)
        key = m.lower()
        if len(m) > 5 and key not in seen:
            seen.add(key)
            cleaned.append(m)
    return "; ".join(cleaned[:12])


def extract_pihak(text: str) -> str:
    patterns = [
        r"Nama\s+lengkap\s*[:\-]?\s*([A-Z][A-Z0-9 .,'`\-]+?)(?=Tempat\s+lahir|Tempat\s+Lahir|Umur|Jenis\s+kelamin|Kebangsaan|Agama|Pekerjaan|;|\d\.)",
        r"Nama\s*[:\-]?\s*([A-Z][A-Z0-9 .,'`\-]+?)(?=Tempat\s+Lahir|Umur|Jenis\s+Kelamin|Kebangsaan|Agama|Pekerjaan|;|\d\.)",
        r"Terdakwa\s*[:\-]?\s*([A-Z][A-Z0-9 .,'`\-]+?)(?=Tempat|Umur|Jenis|Kebangsaan|Agama|Pekerjaan|;|,)",
        r"perkara\s+(?:atas\s+nama\s+)?Terdakwa\s*[:\-]?\s*([A-Z][A-Z0-9 .,'`\-]{3,80})",
    ]
    for pattern in patterns:
        value = _search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        value = re.sub(r"\b(bin|binti)\b", lambda m: m.group(1).lower(), value, flags=re.IGNORECASE)
        value = _compact(value)
        if value and value.lower() not in {"tersebut", "sebagai berikut"}:
            value = re.sub(r"\s+", " ", value)
            return value[:100]
    return ""


def extract_section(text: str, start_patterns: List[str], end_patterns: Optional[List[str]] = None, max_chars: int = 2500, prefer_last: bool = False) -> str:
    lowered = text.lower()
    starts: List[int] = []
    for pattern in start_patterns:
        starts.extend([m.start() for m in re.finditer(pattern, lowered, flags=re.IGNORECASE)])
    if not starts:
        return ""

    start_index = max(starts) if prefer_last else min(starts)
    end_index = len(text)
    if end_patterns:
        search_area = lowered[start_index + 20 :]
        end_candidates = []
        for pattern in end_patterns:
            match = re.search(pattern, search_area, flags=re.IGNORECASE)
            if match:
                end_candidates.append(start_index + 20 + match.start())
        if end_candidates:
            end_index = min(end_candidates)
    section = _compact(text[start_index:end_index])
    return section[:max_chars]


def summarize_facts(text: str) -> str:
    section = extract_section(
        text,
        start_patterns=[r"dakwaan", r"bahwa\s+ia\s+terdakwa", r"bahwa\s+terdakwa", r"pertama\s*:", r"kesatu\s*:"],
        end_patterns=[r"menimbang", r"tuntutan", r"mengingat", r"mengadili"],
        max_chars=1200,
        prefer_last=False,
    )
    if section:
        return section
    return " ".join(text.split()[:180])


def extract_legal_argument(text: str) -> str:
    section = extract_section(
        text,
        start_patterns=[r"menimbang\s*,?\s*bahwa", r"unsur\s+", r"mengingat"],
        end_patterns=[r"mengadili\s*[:\-]", r"demikian\s+diputus"],
        max_chars=1400,
        prefer_last=True,
    )
    return section


def extract_amar(text: str) -> str:
    """Extract the final operative order, not the introductory 'mengadili' in the header."""
    start_candidates = []
    for pattern in [r"MENGADILI\s*[:\-]", r"MENGADILI\s+sendiri\s*[:\-]?"]:
        start_candidates.extend([m.start() for m in re.finditer(pattern, text, flags=re.IGNORECASE)])
    if not start_candidates:
        return ""

    # Prefer a candidate in the last 45 percent of the document. Otherwise use the last candidate.
    threshold = int(len(text) * 0.55)
    late_candidates = [idx for idx in start_candidates if idx >= threshold]
    start_index = late_candidates[-1] if late_candidates else start_candidates[-1]

    end_candidates = []
    for pattern in [r"Demikian\s+diputus", r"Putusan\s+tersebut\s+diucapkan", r"Hakim\s+Ketua", r"Panitera\s+Pengganti"]:
        match = re.search(pattern, text[start_index + 20 :], flags=re.IGNORECASE)
        if match:
            end_candidates.append(start_index + 20 + match.start())
    end_index = min(end_candidates) if end_candidates else min(len(text), start_index + 2200)
    return _compact(text[start_index:end_index])[:1800]


def infer_label_putusan(amar: str, text: str) -> str:
    final = (amar or "").lower()
    whole = (text or "").lower()
    near_end = whole[-7000:]

    # The final amar receives the highest priority because appeal decisions often quote
    # earlier requests, objections, and trial-level reasoning before the operative order.
    if re.search(r"menjatuhkan\s+pidana[^.]{0,250}penjara|pidana\s+penjara\s+selama", final):
        return "pidana penjara"
    if re.search(r"menjatuhkan\s+pidana[^.]{0,250}denda|pidana\s+denda", final):
        return "pidana denda"
    if re.search(r"membebaskan\s+terdakwa|bebas\s+dari\s+segala\s+dakwaan|tidak\s+terbukti\s+secara\s+sah", final):
        return "bebas"
    if re.search(r"melepaskan\s+terdakwa|lepas\s+dari\s+segala\s+tuntutan", final):
        return "lepas"
    if re.search(r"menolak\s+permintaan|permintaan\s+banding[^.]{0,80}ditolak|ditolak", final):
        return "ditolak"
    if re.search(r"menguatkan\s+putusan", final):
        # Try to infer the substantive result from the nearby quoted trial-level amar.
        if re.search(r"menjatuhkan\s+pidana[^.]{0,250}penjara|pidana\s+penjara\s+selama", near_end):
            return "pidana penjara"
        if re.search(r"membebaskan\s+terdakwa|tidak\s+terbukti\s+secara\s+sah", near_end):
            return "bebas"
        return "menguatkan"

    # Fallback for files whose final amar cannot be extracted cleanly.
    target = f"{near_end} {whole[:2500]}"
    if re.search(r"menjatuhkan\s+pidana[^.]{0,250}penjara|pidana\s+penjara\s+selama", target):
        return "pidana penjara"
    if re.search(r"menjatuhkan\s+pidana[^.]{0,250}denda|pidana\s+denda", target):
        return "pidana denda"
    if re.search(r"membebaskan\s+terdakwa|tidak\s+terbukti\s+secara\s+sah|bebas\s+dari\s+segala\s+dakwaan", target):
        return "bebas"
    return "tidak diketahui"

def build_retrieval_text(row: Dict[str, object]) -> str:
    return _compact(
        " ".join(
            [
                str(row.get("jenis_perkara", "")),
                str(row.get("pasal", "")),
                str(row.get("ringkasan_fakta", "")),
                str(row.get("argumen_hukum", "")),
                str(row.get("amar_putusan", "")),
                str(row.get("text_full", ""))[:12000],
            ]
        )
    )


def extract_metadata_from_text(case_id: str, source_file: str, text: str) -> Dict[str, object]:
    amar = extract_amar(text)
    row: Dict[str, object] = {
        "case_id": case_id,
        "source_file": source_file,
        "no_perkara": extract_no_perkara(text),
        "tanggal_putusan": extract_tanggal_putusan(text),
        "pengadilan": extract_pengadilan(text),
        "jenis_perkara": infer_jenis_perkara(text),
        "pasal": extract_pasal(text),
        "pihak": extract_pihak(text),
        "ringkasan_fakta": summarize_facts(text),
        "argumen_hukum": extract_legal_argument(text),
        "amar_putusan": amar,
        "label_putusan": infer_label_putusan(amar, text),
        "word_count": len(text.split()),
        "text_full": text,
    }
    row["retrieval_text"] = build_retrieval_text(row)
    return row


def build_case_representation(raw_dir: Path = RAW_TEXT_DIR) -> pd.DataFrame:
    """Build structured case representation from cleaned text files."""
    ensure_dirs()
    files = sorted(raw_dir.glob("case_*.txt"))
    rows = []
    for file_path in files:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        rows.append(extract_metadata_from_text(file_path.stem, file_path.name, text))

    columns = [
        "case_id", "source_file", "no_perkara", "tanggal_putusan", "pengadilan", "jenis_perkara",
        "pasal", "pihak", "ringkasan_fakta", "argumen_hukum", "amar_putusan", "label_putusan",
        "word_count", "retrieval_text", "text_full",
    ]
    df = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CASES_CSV, index=False)
    df.to_json(CASES_JSON, orient="records", force_ascii=False, indent=2)
    try:
        df.to_excel(PROCESSED_DIR / "cases.xlsx", index=False)
    except Exception:
        pass
    return df


if __name__ == "__main__":
    cases = build_case_representation()
    print(f"Saved {len(cases)} cases to {CASES_CSV}")
