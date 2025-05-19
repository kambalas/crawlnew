#!/usr/bin/env python3
import json, re, unicodedata
from pathlib import Path
# ──────────────────────────────────────────────────────────────────────────────
def canonical_quote(s: str) -> str:
    return re.sub(r'[“”‟«»„]', '"', s)

def load_actual_for_url(path: Path, url_key: str):
    data = load(path)
    if isinstance(data, dict):        # url → list‑of‑products structure
        return data.get(url_key, [])
    if isinstance(data, list):        # already a flat list
        return data
    raise TypeError("Unsupported ACTUAL JSON structure")

def normalize_price(p: str) -> str:
    if not p:
        return ""
    p = p.replace("\u202f", " ")          # narrow NBSP in some euro prices
    number = re.sub(r"[^\d.,]", "", p).replace(",", ".")
    try:
        return f"{float(number):.2f}"
    except ValueError:
        return  number

def normalize_name(n: str) -> str:
    n = unicodedata.normalize("NFKD", n)
    n = canonical_quote(n)
    return re.sub(r"\s+", " ", n).strip().lower()
# ──────────────────────────────────────────────────────────────────────────────
def load(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)

def flatten_actual(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):                       # url → list‑of‑items
        items = []
        for lst in raw.values():
            if isinstance(lst, list):
                items.extend(lst)
        return items
    raise TypeError("Unsupported JSON structure")

# ──────────────────────────────────────────────────────────────────────────────
def evaluate(expected, actual):
    exp_set = {
        (normalize_name(e["name"]), normalize_price(e["price"]))
        for e in expected
    }

    act_set = {
        (normalize_name(a["name"]), normalize_price(a["price"]))
        for a in actual
        if isinstance(a, dict) and not a.get("error")
    }

    matched_act = set()
    tp = 0
    for en, ep in exp_set:
        for idx, (an, ap) in enumerate(act_set):
            if idx in matched_act:
                continue
            if (en and en == an) or (ep and ep == ap):
                tp += 1
                matched_act.add(idx)
                break

    fn = len(exp_set) - tp          # missed ground‑truth items
    fp = len(act_set) - len(matched_act)   # extras we scraped

    
    prec = tp / (tp + fp) * 100 if tp or fp else 0
    rec  = tp / (tp + fn) * 100 if tp or fn else 0
    f1   = 2*prec*rec / (prec + rec) if prec and rec else 0
    print(f"✓ true‑positives : {tp}")
    print(f"✗ false‑negatives: {fn}")
    print(f"⚠ false‑positives: {fp}")
    print(f"\nprecision : {prec:5.2f}%")
    print(f"coverage  : {rec:5.2f}%")      # ← same as recall
    print(f"F1‑score  : {f1:5.2f}%")
    print("-" * 60)

    return tp, fp, fn

# ──────────────────────────────────────────────────────────────────────────────
def main():
    expected = load(Path("archive_expected/mp5.json"))
    url = "https://www.mp.lt/katalogas/ismanieji-telefonai/?filter=brands:223" #



    actual = load_actual_for_url(Path("tmp_results/combined_results.json"), url)

    evaluate(expected, actual)

if __name__ == "__main__":
    main()
