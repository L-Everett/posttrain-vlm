import re

NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
REL_TOL = 0.05


def parse_number(s: str):
    t = s.strip().rstrip(".").replace(",", "")
    is_pct = t.endswith("%")
    if is_pct:
        t = t[:-1]
    try:
        return float(t)
    except ValueError:
        return None


def normalize(s: str) -> str:
    s = s.strip().rstrip(".").lower()
    return re.sub(r"[^a-z0-9%.\- ]", "", s)


def extract_numbers(pred: str):
    out = []
    for m in NUM_RE.findall(pred):
        v = parse_number(m)
        if v is not None:
            out.append(v)
    return out


def score(gold: str, pred: str) -> bool:
    gold_num = parse_number(gold)
    if gold_num is not None:
        nums = extract_numbers(pred)
        for p in nums:
            if gold_num == 0:
                if abs(p) <= 0.05:
                    return True
            elif abs(p - gold_num) / abs(gold_num) <= REL_TOL:
                return True
        return False
    gold_norm = normalize(gold)
    pred_norm = normalize(pred)
    if gold_norm == pred_norm:
        return True
    sentences = re.split(r"[.\n]", pred_norm)
    return any(gold_norm == s.strip() for s in sentences if s.strip())
