
import json, re, unicodedata
from pathlib import Path

# ──────────────────────────────────────────────────────────────
def canonical_quote(s: str) -> str:
    return re.sub(r'[“”‟«»„]', '"', s)

def load_actual_for_url(path: Path, url_key: str):
    data = load(path)
    if isinstance(data, dict):        
        return data.get(url_key, [])
    if isinstance(data, list):       
        return data
    raise TypeError("Unsupported ACTUAL JSON structure")

def normalize_price(p: str) -> str:
    if not p:
        return ""
    p = p.replace("\u202f", " ")          

    number = re.sub(r"[^\d.,]", "", p).replace(",", ".")
    try:
        return f"{float(number):.2f}"
    except ValueError:
        return number

def normalize_name(n: str) -> str:
    n = unicodedata.normalize("NFKD", n)
    n = canonical_quote(n)
    return re.sub(r"\s+", " ", n).strip().lower()

# ──────────────────────────────────────────────────────────────
def load(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)

def flatten_actual(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):                       
        items = []
        for lst in raw.values():
            if isinstance(lst, list):
                items.extend(lst)
        return items
    raise TypeError("Unsupported JSON structure")


def evaluate(expected, actual, quiet=False):
    exp_items = [
        (normalize_name(e["name"]), normalize_price(e["price"]))
        for e in expected
    ]
    act_items = [
        (normalize_name(a["name"]), normalize_price(a["price"]))
        for a in actual
        if isinstance(a, dict) and not a.get("error")
    ]

    matched_act = set()
    tp = 0
    for en, ep in exp_items:
        for idx, (an, ap) in enumerate(act_items):
            if idx in matched_act:
                continue
            if (en and en == an) or (ep and ep == ap):
                tp += 1
                matched_act.add(idx)
                break

    fn = len(exp_items) - tp          
    fp = len(act_items) - len(matched_act)   

    if not quiet:
        prec = tp / (tp + fp) * 100 if tp or fp else 0
        rec  = tp / (tp + fn) * 100 if tp or fn else 0
        f1   = 2*prec*rec / (prec + rec) if prec and rec else 0
        print(f"✓ true‑positives : {tp}")
        print(f"✗ false‑negatives: {fn}")
        print(f"⚠ false‑positives: {fp}")
        print(f"\nprecision : {prec:5.2f}%")
        print(f"coverage  : {rec:5.2f}%")     
        print(f"F1‑score  : {f1:5.2f}%")
        print("-" * 60)

    return tp, fp, fn



# ──────────────────────────────────────────────────────────────
ARCHIVE_DIR   = Path("archive_expected")
COMBINED_JSON = Path("tmp_results/combined_results.json")
URLS_FILE     = Path("urls.txt")
TRUTHS_FILE   = Path("truths.txt")

# ──────────────────────────────────────────────────────────────
def main():
    urls   = [u.strip() for u in URLS_FILE.read_text(encoding="utf-8").splitlines() if u.strip()]
    truths = [t.strip() for t in TRUTHS_FILE.read_text(encoding="utf-8").splitlines() if t.strip()]

    if len(urls) != len(truths):
        raise ValueError(
            f"urls.txt has {len(urls)} lines but truths.txt has {len(truths)} lines "
            "(they must be 1‑to‑1 and in the same order)."
        )

    print(f"Running evaluation for {len(urls)} pages …\n{'='*60}")

    total_tp = total_fp = total_fn = total_found_all = 0


    for url, truth_file in zip(urls, truths):
        expected_path = ARCHIVE_DIR / truth_file
        if not expected_path.exists():
            print(f"‼ expected file not found: {expected_path}")
            print("-" * 60)
            continue

        print(f"URL : {url}")
        print(f"GT  : {truth_file}")

        expected = load(expected_path)
        actual   = load_actual_for_url(COMBINED_JSON, url)

        tp, fp, fn = evaluate(expected, actual)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_found_all += len([
    a for a in actual
    if isinstance(a, dict) and not a.get("error")
])

    # ─────────── overall summary ───────────
    print("\nOVERALL RESULTS\n" + "="*60)
    prec = total_tp / (total_tp + total_fp) * 100 if total_tp or total_fp else 0
    cov  = total_tp / (total_tp + total_fn) * 100 if total_tp or total_fn else 0
    f1   = 2*prec*cov / (prec + cov) if prec and cov else 0

    print(f"✓ total true‑positives : {total_tp}")
    print(f"✗ total false‑negatives: {total_fn}")
    print(f"⚠ total false‑positives: {total_fp}")
    print(f"📦 total found     : {total_found_all}")
    print(f"\nprecision : {prec:5.2f}%")
    print(f"coverage  : {cov:5.2f}%")
    print(f"F1‑score  : {f1:5.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
