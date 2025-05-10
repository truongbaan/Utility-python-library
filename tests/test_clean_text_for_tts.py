import pytest
from freeai_utils.clean_text_for_tts import clean_ai_text_for_tts 

#this is written by AI, will be changed later
def test_type_error():
    with pytest.raises(TypeError):
        clean_ai_text_for_tts(123)

def test_fenced_code_block():
    text = "Here is some code:\n```python\nprint('hello')\n```"
    expected = "Here is some code:"
    assert clean_ai_text_for_tts(text) == expected

def test_inline_code():
    text = "Use `print('hi')` to print."
    expected = "Use print('hi') to print."
    assert clean_ai_text_for_tts(text) == expected

def test_markdown_image():
    text = "This is an image ![alt](image.jpg)"
    expected = "This is an image"
    assert clean_ai_text_for_tts(text) == expected

def test_markdown_link():
    text = "Click [here](http://example.com)"
    expected = "Click here"
    assert clean_ai_text_for_tts(text) == expected

def test_raw_url():
    text = "Visit https://example.com now"
    expected = "Visit  now"
    assert clean_ai_text_for_tts(text) == expected

def test_html_tags():
    text = "This is <b>bold</b> text"
    expected = "This is bold text"
    assert clean_ai_text_for_tts(text) == expected

def test_markdown_formatting():
    text = "This is **bold**, *italic*, and __underline__"
    expected = "This is bold, italic, and underline"
    assert clean_ai_text_for_tts(text) == expected

def test_headings_and_lists():
    text = "# Heading\n- item 1\n2. item 2\n#comment\nText"
    expected = "Heading\nitem 1\nitem 2\n\nText"
    assert clean_ai_text_for_tts(text) == expected

def test_blank_line_collapse():
    text = "Line 1\n\n\nLine 2"
    expected = "Line 1\n\nLine 2"
    assert clean_ai_text_for_tts(text) == expected

def test_strip_whitespace():
    text = "   Trim this text   "
    expected = "Trim this text"
    assert clean_ai_text_for_tts(text) == expected
