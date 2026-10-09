import math
import re

from PIL import Image

INSTRUCTION = "Please answer with a single value."
DEFAULT_MAX_PIXELS = 768 * 768
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
REL_TOL = 0.05


def parse_number(s):
    t = str(s).strip().rstrip(".").replace(",", "")
    if t.endswith("%"):
        t = t[:-1]
    try:
        return float(t)
    except ValueError:
        return None


def normalize(s):
    s = str(s).strip().rstrip(".").lower()
    return re.sub(r"[^a-z0-9%.\- ]", "", s)


def score(gold, pred):
    gold_num = parse_number(gold)
    if gold_num is not None:
        for m in NUM_RE.findall(str(pred)):
            p = parse_number(m)
            if p is None:
                continue
            if gold_num == 0:
                if abs(p) <= REL_TOL:
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


def load_image(path, max_pixels):
    image = Image.open(path).convert("RGB")
    if max_pixels and image.width * image.height > max_pixels:
        factor = math.sqrt(max_pixels / (image.width * image.height))
        image = image.resize((int(image.width * factor), int(image.height * factor)))
    return image
