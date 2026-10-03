"""
Script-aware text normalization shared by every retrieval channel.

Why this exists: the original tokenizers only matched ``[a-zA-Z0-9_-]``, which silently
dropped every Cyrillic letter and every diacritic (č, ć, š, ž, đ, é, ü …). A Serbian
document written in Cyrillic therefore produced *zero* tokens, and Latin-script Serbian
lost letters mid-word ("krađa" -> "kra", "a").

``normalize_text`` maps text to one canonical form so that the index and the query
always meet in the same space:

* lowercase
* Serbian Cyrillic -> Serbian Latin (ђ->dj, љ->lj, њ->nj, џ->dz, ћ/ч->c, ш->s, ж->z)
* Latin diacritics stripped (č/ć->c, š->s, ž->z, đ->dj, é->e, ü->u, ...)

Letters of other scripts (Greek, other Cyrillic letters such as ы/э/я, CJK, ...) are kept
as-is, so non-Serbian text is still tokenized; it simply isn't transliterated.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List

_SR_CYR_TO_LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ђ": "dj", "е": "e", "ж": "z", "з": "z",
    "и": "i", "ј": "j", "к": "k", "л": "l", "љ": "lj", "м": "m", "н": "n", "њ": "nj", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "ћ": "c", "у": "u", "ф": "f", "х": "h", "ц": "c",
    "ч": "c", "џ": "dz", "ш": "s",
}
_TRANSLATE = str.maketrans(_SR_CYR_TO_LAT)

# Word tokens: any run of Unicode letters/digits/underscore, optionally joined by '-'
# (keeps code identifiers such as ``__init__`` or ``vector-store`` intact).
_WORD_RE = re.compile(r"\w[\w\-]*", re.UNICODE)


def normalize_text(text: str) -> str:
    """Lowercase, transliterate Serbian Cyrillic to Latin and strip diacritics."""
    if not text:
        return ""
    t = text.lower().translate(_TRANSLATE)
    # đ / dž have no decomposition in Unicode, handle them explicitly.
    t = t.replace("đ", "dj").replace("dž", "dz")
    t = unicodedata.normalize("NFKD", t)
    return "".join(ch for ch in t if not unicodedata.combining(ch))


def word_tokens(text: str) -> List[str]:
    """Normalized word tokens (Unicode-aware, script-normalized)."""
    return _WORD_RE.findall(normalize_text(text))
