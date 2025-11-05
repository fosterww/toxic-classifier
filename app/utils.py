import re

import emoji
from bs4 import BeautifulSoup

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = BeautifulSoup(s, "html.parser").get_text(separator=" ")
    s = emoji.replace_emoji(s, replace=" <EMOJI> ")
    s = _WS_RE.sub(" ", s).strip().lower()
    return s
