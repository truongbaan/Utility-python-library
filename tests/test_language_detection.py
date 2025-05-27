import pytest
import os
from freeai_utils.language_detection import LangTranslator 

path = os.path.dirname(os.path.abspath(__file__)) + "\\sample"

@pytest.fixture
def translator():
    return LangTranslator()

def get_text():
    try:
        with open('sample\\sample.txt', 'r', encoding='utf-8') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print("The file 'my_file.txt' was not found.")

def test_default(translator):
    content = get_text()
    for line in content:
        origin = translator.translate(line.split("|")[0].strip())
        trans = line.split("|")[1].strip()
        print("Origin: " + origin)
        print("Trans: " + trans)
        assert origin == trans
