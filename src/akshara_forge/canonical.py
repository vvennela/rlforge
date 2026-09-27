"""Conservative, reproducible PDF canonicalization with raw OCR strips.

No OCR engine is treated as a mathematical verifier. Original pages, native text,
OCR TSV, coordinates, and uncertainty flags remain available for source review.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pymupdf as fitz

from .io import digest_file, write_json


def normalize_text(text: str) -> str:
    # NFC preserves superscripts and mathematical distinctions that NFKC loses.
    text = unicodedata.normalize("NFC", text).replace("\x00", "")
    for a, b in {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}.items():
        text = text.replace(a, b)
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()).strip()


def parse_tsv(tsv: str) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        if row.get("level") != "5" or not (row.get("text") or "").strip():
            continue
        try:
            word = {"text": row["text"], "confidence": float(row["conf"]),
                    "bbox": [int(row["left"]), int(row["top"]),
                             int(row["left"]) + int(row["width"]),
                             int(row["top"]) + int(row["height"])]}
        except (ValueError, KeyError, TypeError):
            continue
        key = tuple(row[k] for k in ("block_num", "par_num", "line_num"))
        groups.setdefault(key, []).append(word)
    lines = []
    for key, words in groups.items():
        boxes = [w["bbox"] for w in words]
        lines.append({"id": ":".join(key), "text": " ".join(w["text"] for w in words),
                      "confidence": round(sum(w["confidence"] for w in words) / len(words), 2),
                      "bbox": [min(b[0] for b in boxes), min(b[1] for b in boxes),
                               max(b[2] for b in boxes), max(b[3] for b in boxes)], "words": words})
    return lines


def strip_ranges(height: int, size: int, overlap: int) -> list[tuple[int, int]]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("strip height must be positive and 0 <= overlap < strip height")
    ranges = []
    top = 0
    while top < height:
        bottom = min(height, top + size)
        ranges.append((top, bottom))
        if bottom == height:
            break
        top = bottom - overlap
    return ranges


def _page(pdf: str, index: int, out: str, dpi: int, strip_height: int, overlap: int,
          force_ocr: bool) -> dict:
    out = Path(out)
    prefix = f"page-{index + 1:04d}"
    with fitz.open(pdf) as doc:
        page = doc[index]
        raw = page.get_text("text", sort=True)
        (out / "raw" / f"{prefix}.native.txt").write_text(raw)
        pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
        image_path = out / "pages" / f"{prefix}.png"
        pix.save(image_path)
        scale = dpi / 72
        usable = len(re.findall(r"[A-Za-z]", raw)) >= 80
        method = "native_pdf_text"
        lines = []
        if force_ocr or not usable:
            if not shutil.which("tesseract"):
                raise RuntimeError("Tesseract is required for scanned pages. Install tesseract and rerun.")
            completed = subprocess.run(["tesseract", str(image_path), "stdout", "-l", "eng", "--psm", "3", "tsv"],
                                       check=True, capture_output=True, text=True, timeout=120,
                                       env={**__import__("os").environ, "OMP_THREAD_LIMIT": "1"})
            tsv = completed.stdout
            (out / "raw" / f"{prefix}.ocr.tsv").write_text(tsv)
            lines = parse_tsv(tsv)
            method = "tesseract"
            text = "\n".join(line["text"] for line in lines)
        else:
            for block in page.get_text("dict", sort=True)["blocks"]:
                for line in block.get("lines", []):
                    value = "".join(span["text"] for span in line["spans"])
                    if value.strip():
                        lines.append({"text": value, "bbox": [round(v * scale) for v in line["bbox"]],
                                      "confidence": None})
            text = raw
        (out / "raw" / f"{prefix}.selected.txt").write_text(text)
        strips = []
        for number, (top, bottom) in enumerate(strip_ranges(pix.height, strip_height, overlap), 1):
            strip_name = f"{prefix}-strip-{number:02d}.png"
            clip = fitz.Rect(0, top / scale, page.rect.width, min(page.rect.height, bottom / scale))
            page.get_pixmap(dpi=dpi, clip=clip, colorspace=fitz.csRGB, alpha=False).save(out / "strips" / strip_name)
            strip_lines = [line for line in lines if top <= (line["bbox"][1] + line["bbox"][3]) / 2 < bottom]
            strip_text = normalize_text("\n".join(line["text"] for line in strip_lines))
            (out / "strips" / strip_name.replace(".png", ".txt")).write_text(strip_text)
            strips.append({"image": f"strips/{strip_name}", "text_path": f"strips/{strip_name.replace('.png', '.txt')}",
                           "bbox_pixels": [0, top, pix.width, bottom], "overlap_pixels": overlap})
    confidences = [line["confidence"] for line in lines if line["confidence"] is not None]
    flags = ["mathematical_notation_unverified"]
    if not text.strip():
        flags.append("empty_page")
    if confidences and sum(confidences) / len(confidences) < 75:
        flags.append("low_ocr_confidence")
    if method == "native_pdf_text":
        flags.append("embedded_text_may_be_legacy_ocr")
    normalized = normalize_text(text)
    candidates = []
    for number, line in enumerate(normalized.splitlines(), 1):
        if re.search(r"\b(theorem|lemma|corollary|definition|proposition|proof)\b", line, re.I):
            candidates.append({"line": number, "text": line, "status": "candidate_requires_review"})
    return {"page": index + 1, "image": f"pages/{prefix}.png", "width_pixels": pix.width,
            "height_pixels": pix.height, "method": method, "text": normalized, "raw_text": text,
            "lines": lines, "strips": strips, "flags": flags, "statement_candidates": candidates,
            "ocr_confidence_mean": round(sum(confidences) / len(confidences), 2) if confidences else None}


def canonicalize(pdf: Path, out: Path, *, paper_id: str | None = None, dpi: int = 160,
                 strip_height: int = 800, overlap: int = 80, workers: int = 3,
                 force_ocr: bool = False, refresh: bool = False) -> dict:
    pdf = pdf.resolve()
    paper_id = paper_id or pdf.stem
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    settings = {"dpi": dpi, "strip_height": strip_height, "overlap": overlap, "force_ocr": force_ocr,
                "normalization": "NFC_ligatures_whitespace_v1"}
    strip_ranges(1, strip_height, overlap)
    source_hash = digest_file(pdf)
    fingerprint = hashlib.sha256(json.dumps({"source": source_hash, **settings}, sort_keys=True).encode()).hexdigest()
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and not refresh:
        prior = json.loads(manifest_path.read_text())
        if prior.get("fingerprint") == fingerprint and (out / "document.json").is_file():
            return {**prior, "cached": True}
    for folder in ("raw", "pages", "strips"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf) as doc:
        count = len(doc)
        metadata = doc.metadata
    pages = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(_page, str(pdf), i, str(out), dpi, strip_height, overlap, force_ocr) for i in range(count)]
        for future in as_completed(jobs):
            result = future.result()
            pages.append(result)
            write_json(out / "pages" / f"page-{result['page']:04d}.json", result)
            print(f"{paper_id}: canonicalized page {result['page']}/{count}", flush=True)
    pages.sort(key=lambda p: p["page"])
    # Repeated headers/footers are excluded from canonical text only; raw text stays intact.
    boundary_counts = Counter()
    for page in pages:
        lines = page["text"].splitlines()
        boundary_counts.update(set(lines[:2] + lines[-2:]))
    repeated = {line for line, n in boundary_counts.items() if n >= max(3, count // 3) and len(line) < 100}
    for page in pages:
        original_lines = page["text"].splitlines()
        page["removed_marginal_lines"] = [line for i, line in enumerate(original_lines)
                                         if (i < 2 or i >= len(original_lines) - 2) and line in repeated]
        page["text"] = "\n".join(line for i, line in enumerate(original_lines)
                                  if not ((i < 2 or i >= len(original_lines) - 2) and line in repeated))
    document = {"schema_version": "1.0", "paper_id": paper_id, "source_pdf": str(pdf),
                "source_sha256": source_hash, "metadata": metadata, "page_count": count,
                "settings": settings, "pages": pages,
                "status": "canonicalized_not_mathematically_verified",
                "notes": "Statement detection is a retrieval aid, not theorem extraction. Review scanned equations before task generation."}
    write_json(out / "document.json", document)
    md = [f"# {paper_id}", "", "Canonical OCR transcription. Mathematical notation remains unverified.", ""]
    for page in pages:
        md += [f"## PDF page {page['page']}", f"Source image: {page['image']}", "", page["text"], ""]
    (out / "document.md").write_text("\n".join(md))
    manifest = {"schema_version": "1.0", "paper_id": paper_id, "fingerprint": fingerprint,
                "source_sha256": source_hash, "page_count": count,
                "methods": dict(Counter(p["method"] for p in pages)),
                "strips": sum(len(p["strips"]) for p in pages), "settings": settings,
                "flagged_pages": [{"page": p["page"], "flags": p["flags"]} for p in pages],
                "document_sha256": digest_file(out / "document.json"), "cached": False}
    write_json(manifest_path, manifest)
    return manifest
