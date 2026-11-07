"""English text frontend: normalisation (numbers, times, years) and lexicon + g2p_en pronunciation."""
import re
import unicodedata
from string import punctuation

from text.numbers import normalize_numbers  # from the reference repo (inflect based)

_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\s*(a\.?m\.?|p\.?m\.?)?", re.I)
_ABBR = {"am": "ay em", "pm": "pee em"}


def _num(n):
    import inflect
    w = inflect.engine().number_to_words(n, andword="")
    return w.replace("-", " ").replace(",", " ")


def _time(m):
    h, mi, ap = int(m.group(1)), int(m.group(2)), m.group(3)
    words = _num(h)
    if mi:
        words += (" oh " if mi < 10 else " ") + _num(mi)
    if ap:
        words += " " + _ABBR[ap.replace(".", "").lower()]
    return words


def normalize(text, keep_punct=False):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = _TIME_RE.sub(_time, text)
    text = normalize_numbers(text)
    text = text.lower().replace("-", " ")
    allowed = r"[^a-z' ,.;:?!]" if keep_punct else r"[^a-z' ]"
    text = re.sub(allowed, " ", text)
    if keep_punct:
        text = re.sub(r"([,.;:?!])", r" \1 ", text)
    text = re.sub(r"(^|\s)'+|'+(\s|$)", " ", text)   # stray quote marks
    return re.sub(r"\s+", " ", text).strip()


def read_lexicon(path):
    lex = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = re.split(r"\s+", line.strip())
            if len(parts) < 2:
                continue
            w = parts[0].lower()
            if w not in lex:
                lex[w] = parts[1:]
    return lex


_PH_RE = re.compile(r"^[A-Z]+[0-2]?$")


class Frontend:
    def __init__(self, lexicon_path="lexicon/librispeech-lexicon.txt"):
        from g2p_en import G2p
        self.lex = read_lexicon(lexicon_path)
        self.g2p = G2p()

    def word_phones(self, w):
        if w in self.lex:
            return list(self.lex[w])
        return [p for p in self.g2p(w) if _PH_RE.match(p)]

    def __call__(self, text):
        norm = normalize(text, keep_punct=True)
        phones = []
        for tok in norm.split():
            if tok in ",.;:?!":
                phones.append("sp")
            else:
                # NOTE: no "sp" between ordinary words. MFA training sequences only contain "sp" at real pauses,
                # so (like ming024/FastSpeech2) we only insert it at punctuation. The previous version put a pause
                # token after EVERY word, which the model had never seen in training.
                phones += self.word_phones(tok)
        while phones and phones[-1] == "sp":
            phones.pop()
        return norm, "{" + " ".join(phones) + "}"

    def build_dictionary(self, words):
        d = {}
        for w in words:
            p = self.word_phones(w)
            if p:
                d[w] = p
        return d


def clean_for_corpus(text):
    return normalize(text, keep_punct=False)
