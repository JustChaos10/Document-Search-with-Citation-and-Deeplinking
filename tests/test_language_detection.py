from app.nlp import detect_language


def test_detect_language_english():
    text = "This is a simple English sentence about data processing."
    assert detect_language(text) == "en"


def test_detect_language_arabic():
    text = "هذه جملة عربية بسيطة تتحدث عن معالجة البيانات."
    assert detect_language(text) == "ar"
