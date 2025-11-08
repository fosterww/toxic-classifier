from app.utils import clean_text


def test_clean_text_basic():
    s = "  Hello   WORLD  "
    out = clean_text(s)
    assert out == "hello world"


def test_clean_text_html():
    s = "<b>Hi</b> there"
    out = clean_text(s)
    assert "hi" in out
    assert "<" not in out and ">" not in out


def test_clean_text_emoji():
    s = "nice 😊"
    out = clean_text(s)
    assert "<emoji>" in out.lower() or "<emoji" in out.upper() or "<emoji>" in out
