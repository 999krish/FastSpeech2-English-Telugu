"""Telugu text frontend: normalisation, number expansion and rule-based grapheme->phone conversion.

Telugu orthography is (almost) phonemic, so a deterministic akshara parser is used instead of a G2P model:
  * every consonant letter            -> one consonant phone (the letter itself)
  * independent vowels / vowel signs  -> one vowel phone (independent-vowel letter)
  * inherent vowel 'a' (అ)            -> inserted after a consonant that has no vowel sign and no virama
  * virama (్)                         -> suppresses the inherent vowel (no phone)
  * anusvara/visarga/candrabindu      -> own phone
Phones are single Unicode characters, so they double as model symbols (see symbols patch).
"""
import re
import unicodedata

VIRAMA = "\u0C4D"
CONS = {chr(c) for c in range(0x0C15, 0x0C3A)} | {"\u0C58", "\u0C59", "\u0C5A"}
INDEP = {chr(c) for c in range(0x0C05, 0x0C15)}
SIGN2VOWEL = {
    "\u0C3E": "\u0C06", "\u0C3F": "\u0C07", "\u0C40": "\u0C08", "\u0C41": "\u0C09",
    "\u0C42": "\u0C0A", "\u0C43": "\u0C0B", "\u0C44": "\u0C60", "\u0C46": "\u0C0E",
    "\u0C47": "\u0C0F", "\u0C48": "\u0C10", "\u0C4A": "\u0C12", "\u0C4B": "\u0C13",
    "\u0C4C": "\u0C14",
}
NASALS = {"\u0C01", "\u0C02", "\u0C03"}
INHERENT = "\u0C05"  # అ
TELUGU_DIGITS = {chr(0x0C66 + i): str(i) for i in range(10)}

# ---------------------------------------------------------------- numbers
_ONES = ["సున్న", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది",
         "పది", "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు",
         "పద్దెనిమిది", "పంతొమ్మిది"]
_TENS = {2: "ఇరవై", 3: "ముప్పై", 4: "నలభై", 5: "యాభై", 6: "అరవై", 7: "డెబ్బై", 8: "ఎనభై", 9: "తొంభై"}


def _below_100(n):
    if n < 20:
        return _ONES[n]
    t, o = divmod(n, 10)
    return _TENS[t] if o == 0 else _TENS[t] + " " + _ONES[o]


def number_to_words(n):
    """Integer -> Telugu words (0..99,999; larger numbers are read digit by digit)."""
    if n >= 100000:
        return " ".join(_ONES[int(d)] for d in str(n))
    if n < 100:
        return _below_100(n)
    parts = []
    th, rem = divmod(n, 1000)
    if th:
        if th == 1:
            parts.append("వెయ్యి")
        else:
            parts.append(_below_100(th) + (" వేలు" if rem == 0 else " వేల"))
    h, rem = divmod(rem, 100)
    if h:
        if h == 1:
            parts.append("వంద" if rem == 0 else "నూట")
        else:
            parts.append(_ONES[h] + (" వందలు" if rem == 0 else " వందల"))
    if rem:
        parts.append(_below_100(rem))
    return " ".join(parts)


def ordinal_suffix(words):
    """'పన్నెండు' + 'వ' -> 'పన్నెండవ' (simple rule: drop final vowel sign, add వ)."""
    if words.endswith("వై"):
        return words[:-1] + "య్యవ"
    if words and words[-1] in "ుి":
        words = words[:-1]
    return words + "వ"


_TIME_RE = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)")
_ORD_RE = re.compile(r"(\d+)\s*వ(?=\s|$|[,.;?!])")
_NUM_RE = re.compile(r"\d+")


def expand_numbers(text):
    text = "".join(TELUGU_DIGITS.get(c, c) for c in text)

    def _time(m):
        h, mi = int(m.group(1)), int(m.group(2))
        w = number_to_words(h)
        if mi:
            w += " " + number_to_words(mi)
        return w
    text = _TIME_RE.sub(_time, text)
    text = _ORD_RE.sub(lambda m: ordinal_suffix(number_to_words(int(m.group(1)))), text)
    text = _NUM_RE.sub(lambda m: number_to_words(int(m.group(0))), text)
    return text


# ---------------------------------------------------------------- normalisation / phones
_PUNCT_RE = re.compile(r"[,;:.!?\u0964\u0965]")


def normalize(text, keep_punct=False):
    """NFC, drop ZWJ/ZWNJ, expand numbers, keep Telugu-block characters only."""
    text = unicodedata.normalize("NFC", text).replace("\u200c", "").replace("\u200d", "")
    text = expand_numbers(text)
    out = []
    for ch in text:
        if "\u0C00" <= ch <= "\u0C7F" and not ("\u0C66" <= ch <= "\u0C6F"):
            out.append(ch)
        elif keep_punct and ch in ",;:.!?":
            out.append(" " + ch + " ")
        else:
            out.append(" ")
    return re.sub(r"\s+", " ", "".join(out)).strip()


def word_to_phones(word):
    phones, i, n = [], 0, len(word)
    while i < n:
        c = word[i]
        if c in CONS:
            phones.append(c)
            nxt = word[i + 1] if i + 1 < n else ""
            if nxt == VIRAMA:
                i += 2
                continue
            if nxt in SIGN2VOWEL:
                phones.append(SIGN2VOWEL[nxt])
                i += 2
                continue
            phones.append(INHERENT)
            i += 1
        elif c in INDEP or c in NASALS:
            phones.append(c)
            i += 1
        elif c in SIGN2VOWEL:          # stray vowel sign
            phones.append(SIGN2VOWEL[c])
            i += 1
        else:                           # stray virama / unknown mark
            i += 1
    return phones


class Frontend:
    """Callable: raw text -> (normalised text, '{ph ph sp ph ...}' string)."""

    def __init__(self, *_, **__):
        pass

    def __call__(self, text):
        norm = normalize(text, keep_punct=True)
        phones = []
        for tok in norm.split():
            if _PUNCT_RE.fullmatch(tok):
                phones.append("sp")
            else:
                if phones and phones[-1] != "sp":
                    phones.append("sp")
                phones += word_to_phones(tok)
        while phones and phones[-1] == "sp":
            phones.pop()
        return norm, "{" + " ".join(phones) + "}"

    def build_dictionary(self, words):
        return build_dictionary(words)


def build_dictionary(words):
    """MFA dictionary lines (word + phones) for a list of normalised words."""
    return {w: word_to_phones(w) for w in words if word_to_phones(w)}


def clean_for_corpus(text):
    return normalize(text, keep_punct=False)
