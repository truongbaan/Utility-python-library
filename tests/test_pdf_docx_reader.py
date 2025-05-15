import pytest
import os
from freeai_utils.pdf_docx_reader import PDF_DOCX_Reader 

path = os.path.dirname(os.path.abspath(__file__)) + "\\sample"

@pytest.fixture
def reader():
    return PDF_DOCX_Reader()

def test_pdf_reading(reader):
    raw = reader.extract_all_text(os.path.join(path, "sample.pdf"))
    expected = "Hi, this is a docx in freeai-utils to check its function.\nThis second line is used to check the functions that able to separate lines."
    norm = "\n".join(line.rstrip() for line in raw.splitlines())
    assert norm == expected
    
    raw = reader.extract_ordered_text(os.path.join(path, "sample2.pdf"))
    expected = "Hi, this is a docx in freeai-utils to check its function.\nThis second line is used to check the functions that able to separate lines."
    norm = "\n".join(line.rstrip() for line in raw.splitlines())
    assert norm == expected


def test_docx_reading(reader):
    text = reader.extract_all_text(os.path.join(path, "sample.docx"))
    expected = "Hi, this is a docx in freeai-utils to check its function.\nThis second line is used to check the functions that able to separate lines."
    print(repr(text))
    print(repr(expected))

    assert text == expected
    
    text = reader.extract_ordered_text(os.path.join(path, "sample2.docx"))
    expected = "Hi, this is a docx in freeai-utils to check its function.\nThis second line is used to check the functions that able to separate lines."
    print(repr(text))
    print(repr(expected))

    assert text == expected

