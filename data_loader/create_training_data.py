"""
data_loader/create_training_data.py

Unified builder: merges Aariz, ISBI 2015 and DentalCepha-102 into one master index
(training_data/master_dataset_index.json) plus a folder of normalised 8-bit grayscale PNGs.

Changes vs. the previous version
  * ISBI_19_INDEX_TO_SYMBOL fixed (9 of 19 entries were wrong; DentalCepha inherited the bug).
  * ISBI split is taken from the folder the image lives in (Training / Test1 / Test2), not from the file number.
  * One shared build_record() instead of three copy-pasted blocks.
  * Pixel size is flagged (pixel_size_verified) instead of silently assumed.
  * Images are always re-written (no stale files from an old run) and stored as 1-channel PNG.
  * Per-landmark inter-annotator distance stored -> gives the human-level error floor in mm.
  * Exact-duplicate detection (image hash) across splits/sources.
  * QC report is computed from the data, not hard-coded.
  * Optional per-source masking of landmarks whose definition differs (lip points).
"""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
from tqdm import tqdm

Point = Tuple[float, float]
PointDict = Dict[str, Point]

# ----------------------------------------------------------------------------
# Landmark definitions
# ----------------------------------------------------------------------------
CANONICAL_29_SYMBOLS = [
    "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Pn", "R",
    "S", "Ar", "Co", "Gn", "Go", "Po", "LPM", "LIT", "LMT", "UPM",
    "UIA", "UIT", "UMT", "LIA", "Li", "Ls", "N`", "Pog`", "Sn",
]

# ISBI 2015 order L1..L19 (also used by the DentalCepha-102 txt files):
# sella, nasion, orbitale, porion, subspinale, supramentale, pogonion, menton, gnathion, gonion,
# lower incisal incision, upper incisal incision, upper lip, lower lip, subnasale,
# soft-tissue pogonion, PNS, ANS, articulare
ISBI_19_INDEX_TO_SYMBOL = {
    1: "S", 2: "N", 3: "Or", 4: "Po", 5: "A", 6: "B", 7: "Pog",
    8: "Me", 9: "Gn", 10: "Go", 11: "LIT", 12: "UIT",
    13: "Ls", 14: "Li", 15: "Sn", 16: "Pog`",
    17: "PNS", 18: "ANS", 19: "Ar",
}
ISBI_SYMBOLS = set(ISBI_19_INDEX_TO_SYMBOL.values())

# Fail at import time if the tables are inconsistent.
assert len(CANONICAL_29_SYMBOLS) == 29 and len(set(CANONICAL_29_SYMBOLS)) == 29
assert len(ISBI_19_INDEX_TO_SYMBOL) == 19 and len(ISBI_SYMBOLS) == 19
assert ISBI_SYMBOLS <= set(CANONICAL_29_SYMBOLS)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
ISBI_PIXEL_SIZE_MM = 0.1
ISBI_EXPECTED_SIZE = (1935, 2400)          # (width, height)

# The PKU 102-image set is documented at 0.096 mm and 2089 x 1937. If your images do not have
# that size, the pixel size is flagged as unverified (see QC report).
CEPHA102_PIXEL_SIZE_MM = 0.096
CEPHA102_EXPECTED_SIZE = (2089, 1937)

# ISBI / DentalCepha "upper lip" and "lower lip" are lip-tip points, Aariz Ls / Li are labrale points.
# If validation shows a per-source bias on them, set e.g. {"ISBI2015": {"Ls", "Li"}, ...}
MASKED_SYMBOLS_BY_SOURCE: Dict[str, Set[str]] = {
    "Aariz": set(),
    "ISBI2015": set(),
    "DentalCepha102": set(),
}

IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
ISBI_FOLDER_TO_SPLIT = {"TrainingData": "train", "Test1Data": "valid", "Test2Data": "test"}


# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
def find_data_root() -> Tuple[Path, Path]:
    """Detect the project root and the raw dataset root."""
    project_root = Path(__file__).resolve().parent.parent
    candidates = [
        project_root / "data",
        project_root.parent / "data",
        project_root / "ALL - Data set",
        project_root.parent / "ALL - Data set",
        Path.cwd() / "data",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return project_root, candidate
    raise FileNotFoundError("Could not find the dataset root ('data' or 'ALL - Data set').")


def find_image_for_stem(image_dir: Path, stem: str) -> Optional[Path]:
    """First image file called <stem>.<ext> in image_dir, or None."""
    if not image_dir.is_dir():
        return None
    for candidate in sorted(image_dir.glob(f"{stem}.*")):
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTS:
            return candidate
    return None


# ----------------------------------------------------------------------------
# Readers
# ----------------------------------------------------------------------------
def read_gray(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Corrupted / unreadable image: {path}")
    return img


def load_machine_metadata(csv_path: Path) -> Dict[str, Dict[str, object]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Mappings CSV not found: {csv_path}")
    meta: Dict[str, Dict[str, object]] = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"cephalogram_id", "machine", "pixel_size", "mode"}
        missing_cols = required - set(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(f"{csv_path.name}: missing columns {missing_cols}; found {reader.fieldnames}")
        for row in reader:
            cid = row["cephalogram_id"].strip()
            if cid in meta:
                raise ValueError(f"{csv_path.name}: duplicate cephalogram_id '{cid}'")
            ps = float(row["pixel_size"])
            if not (0.01 < ps < 1.0):
                raise ValueError(f"{csv_path.name}: implausible pixel_size {ps} for '{cid}'")
            meta[cid] = {"machine": row["machine"].strip(), "pixel_size": ps, "mode": row["mode"].strip()}
    return meta


def load_aariz_json(json_path: Path, ceph_id: str) -> PointDict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pts: PointDict = {}
    for item in data.get("landmarks", []):
        sym = item.get("symbol")
        val = item.get("value", {})
        if sym and "x" in val and "y" in val:
            if sym in pts:
                raise ValueError(f"[Duplicate Landmark] {ceph_id}: '{sym}' appears twice in {json_path.name}")
            pts[sym] = (float(val["x"]), float(val["y"]))
    missing = set(CANONICAL_29_SYMBOLS) - set(pts)
    extra = set(pts) - set(CANONICAL_29_SYMBOLS)
    if missing or extra:
        raise ValueError(f"[Landmark Mismatch] {ceph_id}: missing={missing}, extra={extra}")
    return pts


def load_isbi_txt(txt_path: Path) -> PointDict:
    """ISBI-style txt: first 19 lines are 'x,y'. Extra lines (if any) are ignored."""
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 19:
        raise ValueError(f"[Corrupt TXT] {txt_path}: expected >= 19 lines, got {len(lines)}")
    pts: PointDict = {}
    for idx in range(1, 20):
        parts = lines[idx - 1].split(",")
        if len(parts) < 2:
            raise ValueError(f"[Invalid Format] {txt_path} line {idx}: '{lines[idx - 1]}'")
        pts[ISBI_19_INDEX_TO_SYMBOL[idx]] = (float(parts[0]), float(parts[1]))
    return pts


# ----------------------------------------------------------------------------
# Merging / checks
# ----------------------------------------------------------------------------
def merge_annotators(a: PointDict, b: PointDict) -> Tuple[PointDict, Dict[str, Optional[float]]]:
    """Mean of the two annotators; inter-annotator distance in pixels (None when only one exists)."""
    merged: PointDict = {}
    disagreement: Dict[str, Optional[float]] = {}
    for sym in CANONICAL_29_SYMBOLS:
        pa, pb = a.get(sym), b.get(sym)
        if pa is not None and pb is not None:
            merged[sym] = ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)
            disagreement[sym] = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
        elif pa is not None or pb is not None:
            merged[sym] = pa if pa is not None else pb
            disagreement[sym] = None
    return merged, disagreement


def assert_in_bounds(record_id: str, pts: PointDict, w: int, h: int) -> None:
    for sym, (x, y) in pts.items():
        if not (0.0 <= x < float(w) and 0.0 <= y < float(h)):
            raise ValueError(f"[Bounds] {record_id}: landmark '{sym}' at ({x:.1f}, {y:.1f}) outside {w}x{h}")


def size_matches(w: int, h: int, expected: Optional[Tuple[int, int]]) -> bool:
    return expected is not None and ((w, h) == expected or (h, w) == expected)


# ----------------------------------------------------------------------------
# Record construction (shared by all sources)
# ----------------------------------------------------------------------------
def build_record(
    *,
    uid: str,
    source: str,
    split: str,
    machine: str,
    mode: str,
    pixel_size: float,
    pixel_size_from_metadata: bool,
    expected_size: Optional[Tuple[int, int]],
    img_src: Path,
    pts_a: PointDict,
    pts_b: PointDict,
    out_images_dir: Path,
) -> dict:
    img = read_gray(img_src)
    h, w = img.shape

    merged, disagreement = merge_annotators(pts_a, pts_b)
    assert_in_bounds(uid, merged, w, h)

    dest_name = f"{uid}.png"
    if not cv2.imwrite(str(out_images_dir / dest_name), img):
        raise IOError(f"Could not write {out_images_dir / dest_name}")

    masked = MASKED_SYMBOLS_BY_SOURCE.get(source, set())
    landmarks = []
    for sym in CANONICAL_29_SYMBOLS:
        if sym in merged and sym not in masked:
            x, y = merged[sym]
            d = disagreement[sym]
            landmarks.append({"symbol": sym, "x": x, "y": y, "valid": True,
                              "inter_annotator_px": None if d is None else round(d, 4)})
        else:
            landmarks.append({"symbol": sym, "x": 0.0, "y": 0.0, "valid": False, "inter_annotator_px": None})

    return {
        "unique_id": uid,
        "source_dataset": source,
        "split": split,
        "machine": machine,
        "acquisition_mode": mode,
        "pixel_size": pixel_size,
        "pixel_size_verified": bool(pixel_size_from_metadata or size_matches(w, h, expected_size)),
        "num_annotators": 2 if pts_b else 1,
        "image_relative_path": f"images/{dest_name}",
        "image_md5": hashlib.md5(img.tobytes()).hexdigest(),
        "orig_width": w,
        "orig_height": h,
        "landmarks": landmarks,
    }


# ----------------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------------
def process_aariz(aariz_root: Path, out_images_dir: Path) -> List[dict]:
    meta = load_machine_metadata(aariz_root / "cephalogram_machine_mappings.csv")
    records: List[dict] = []
    for split in ("train", "valid", "test"):
        ann = aariz_root / split / "Annotations" / "Cephalometric Landmarks"
        senior_dir, junior_dir = ann / "Senior Orthodontists", ann / "Junior Orthodontists"
        images_dir = aariz_root / split / "Cephalograms"

        json_files = sorted(senior_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No senior annotation files in {senior_dir}")

        for s_json in tqdm(json_files, desc=f"Aariz {split}"):
            cid = s_json.stem
            if cid not in meta:
                raise KeyError(f"Aariz {cid}: not found in cephalogram_machine_mappings.csv")
            img_src = find_image_for_stem(images_dir, cid)
            if img_src is None:
                raise FileNotFoundError(f"Missing Aariz image for {cid} in {images_dir}")

            j_json = junior_dir / f"{cid}.json"
            records.append(build_record(
                uid=f"aariz_{cid}", source="Aariz", split=split,
                machine=meta[cid]["machine"], mode=meta[cid]["mode"],
                pixel_size=meta[cid]["pixel_size"], pixel_size_from_metadata=True,
                expected_size=None, img_src=img_src,
                pts_a=load_aariz_json(s_json, cid),
                pts_b=load_aariz_json(j_json, cid) if j_json.exists() else {},
                out_images_dir=out_images_dir,
            ))
    return records


def process_isbi(raw_root: Path, out_images_dir: Path) -> List[dict]:
    isbi_dir = next((c for c in [raw_root / "3471833", raw_root / "Dataset3",
                                 raw_root / "Dataset 3", raw_root / "dataset3"] if c.exists()), None)
    if isbi_dir is None:
        print(f"[*] Skipping ISBI: no matching folder under {raw_root}")
        return []

    senior_dir = isbi_dir / "AnnotationsByMD" / "400_senior"
    junior_dir = isbi_dir / "AnnotationsByMD" / "400_junior"
    raw_img_dir = isbi_dir / "RawImage"

    txt_files = sorted(senior_dir.glob("*.txt"))
    if not txt_files:
        print(f"[*] Skipping ISBI: no annotation files in {senior_dir}")
        return []

    records: List[dict] = []
    for s_txt in tqdm(txt_files, desc="ISBI 400"):
        num = s_txt.stem
        img_src, split = None, None
        for folder, sp in ISBI_FOLDER_TO_SPLIT.items():
            cand = find_image_for_stem(raw_img_dir / folder, num)
            if cand is not None:
                img_src, split = cand, sp
                break
        if img_src is None:
            raise FileNotFoundError(f"Missing ISBI image for sample {num}")

        j_txt = junior_dir / f"{num}.txt"
        records.append(build_record(
            uid=f"isbi_{num}", source="ISBI2015", split=split,
            machine="ISBI_Standard", mode="unknown",
            pixel_size=ISBI_PIXEL_SIZE_MM, pixel_size_from_metadata=False,
            expected_size=ISBI_EXPECTED_SIZE, img_src=img_src,
            pts_a=load_isbi_txt(s_txt),
            pts_b=load_isbi_txt(j_txt) if j_txt.exists() else {},
            out_images_dir=out_images_dir,
        ))
    return records


def process_cepha102(raw_root: Path, out_images_dir: Path) -> List[dict]:
    root = next((c for c in [raw_root / "dental-cepha-dataset",
                             raw_root / "dataset 2" / "dental-cepha-dataset",
                             raw_root / "dataset 2"] if c.exists()), None)
    if root is None:
        print(f"[*] Skipping DentalCepha: no matching folder under {raw_root}")
        return []

    doc1_dir, doc2_dir, img_dir = root / "doctor1", root / "doctor2", root / "image"
    txt_files = sorted(doc1_dir.glob("*.txt"))
    if not txt_files:
        print(f"[*] Skipping DentalCepha: no txt files in {doc1_dir}")
        return []

    records: List[dict] = []
    for d1 in tqdm(txt_files, desc="DentalCepha 102"):
        name = d1.stem
        img_src = find_image_for_stem(img_dir, name)
        if img_src is None:
            raise FileNotFoundError(f"Missing DentalCepha image for {name}")
        d2 = doc2_dir / f"{name}.txt"
        records.append(build_record(
            uid=f"cepha102_{name}", source="DentalCepha102", split="train",
            machine="DentalCepha_Standard", mode="unknown",
            pixel_size=CEPHA102_PIXEL_SIZE_MM, pixel_size_from_metadata=False,
            expected_size=CEPHA102_EXPECTED_SIZE, img_src=img_src,
            pts_a=load_isbi_txt(d1),
            pts_b=load_isbi_txt(d2) if d2.exists() else {},
            out_images_dir=out_images_dir,
        ))
    return records


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def build_report(records: List[dict]) -> dict:
    by_split = Counter((r["source_dataset"], r["split"]) for r in records)
    sizes = defaultdict(Counter)
    for r in records:
        sizes[r["source_dataset"]][f'{r["orig_width"]}x{r["orig_height"]}'] += 1

    unverified = Counter(r["source_dataset"] for r in records if not r["pixel_size_verified"])
    single = Counter(r["source_dataset"] for r in records if r["num_annotators"] == 1)
    masked = Counter()
    for r in records:
        masked[r["source_dataset"]] += sum(1 for lm in r["landmarks"] if not lm["valid"])

    by_hash = defaultdict(list)
    for r in records:
        by_hash[r["image_md5"]].append(r)
    dup_groups = [[(x["unique_id"], x["split"]) for x in g] for g in by_hash.values() if len(g) > 1]
    leaking = [g for g in dup_groups if len({sp for _, sp in g}) > 1]

    return {
        "total": len(records),
        "counts": {f"{s}/{sp}": n for (s, sp), n in sorted(by_split.items())},
        "top_image_sizes": {s: c.most_common(3) for s, c in sizes.items()},
        "pixel_size_unverified": dict(unverified),
        "records_with_single_annotator": dict(single),
        "invalid_landmark_slots": dict(masked),
        "exact_duplicate_groups": dup_groups,
        "duplicates_across_splits": leaking,
    }


def print_report(report: dict, images_dir: Path, master_json: Path) -> None:
    line = "=" * 64
    print("\n" + line)
    print("  DATASET UNIFICATION & QC REPORT")
    print(line)
    print(f" Total samples: {report['total']}")
    for k, v in report["counts"].items():
        print(f"   {k:<26} {v}")
    print(" Most common image sizes per source:")
    for s, c in report["top_image_sizes"].items():
        print(f"   {s:<16} {c}")
    print(f" Single-annotator records:   {report['records_with_single_annotator'] or 'none'}")
    print(f" Invalid landmark slots:     {report['invalid_landmark_slots']}")
    print(" Bounds check:               passed for all landmarks (script aborts otherwise)")
    if report["pixel_size_unverified"]:
        print(f" WARNING pixel size unverified for: {report['pixel_size_unverified']}")
        print("   -> confirm the spacing, or exclude these images from mm-based metrics.")
    else:
        print(" Pixel size:                 verified for all sources")
    n_dup, n_leak = len(report["exact_duplicate_groups"]), len(report["duplicates_across_splits"])
    print(f" Exact duplicate groups:     {n_dup} ({n_leak} span more than one split)")
    if n_leak:
        print("   WARNING train/val/test leakage:")
        for g in report["duplicates_across_splits"][:10]:
            print(f"     {g}")
    print(f" Images:       {images_dir}")
    print(f" Master index: {master_json}")
    print(line + "\n")


# ----------------------------------------------------------------------------
def main() -> None:
    project_root, raw_root = find_data_root()
    print(f"[*] Project root: {project_root}")
    print(f"[*] Raw data root: {raw_root}")

    out_dir = project_root / "training_data"
    out_images_dir = out_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    aariz_root = raw_root / "Aariz"
    if not aariz_root.is_dir():
        raise FileNotFoundError(f"Aariz directory not found: {aariz_root}")

    records: List[dict] = []
    print("\n[1/3] Aariz")
    records += process_aariz(aariz_root, out_images_dir)
    print("\n[2/3] ISBI 2015")
    records += process_isbi(raw_root, out_images_dir)
    print("\n[3/3] DentalCepha 102")
    records += process_cepha102(raw_root, out_images_dir)

    ids = [r["unique_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("unique_id collision between sources")

    report = build_report(records)

    master_json = out_dir / "master_dataset_index.json"
    with open(master_json, "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": 2,
            "canonical_landmarks_29": CANONICAL_29_SYMBOLS,
            "isbi_index_to_symbol": ISBI_19_INDEX_TO_SYMBOL,
            "total_samples": len(records),
            "samples": records,
        }, f, indent=1)
    with open(out_dir / "build_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)

    print_report(report, out_images_dir, master_json)


if __name__ == "__main__":
    main()