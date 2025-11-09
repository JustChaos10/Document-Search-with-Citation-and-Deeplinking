from app.web.routes import _make_card_snippet


def test_snippet_removes_leading_this_part():
    snippet = _make_card_snippet("This part: contains meaningful content about research progress.")
    assert not snippet.lower().startswith("this part")


def test_snippet_truncates_cleanly():
    text = " ".join(["analysis"] * 80)
    snippet = _make_card_snippet(text)
    assert snippet.endswith("…")
    assert "  " not in snippet
