import pytest
import os
from freeai_utils.pdf_docx_reader import PDF_DOCX_Reader 

path = os.path.dirname(os.path.abspath(__file__)) + "\\sample"

@pytest.fixture
def reader():
    return PDF_DOCX_Reader()

def test_pdf_reading_s1(reader):
    raw = reader.extract_all_text(os.path.join(path, "sample.pdf"))
    expected = "Hi, this is a docx in freeai-utils to check its function.\nThis second line is used to check the functions that able to separate lines."
    norm = "\n".join(line.rstrip() for line in raw.splitlines())
    assert norm == expected
    
    raw = reader.extract_ordered_text(os.path.join(path, "sample.pdf"))
    expected = "Hi, this is a docx in freeai-utils to check its function.\nThis second line is used to check the functions that able to separate lines."
    norm = "\n".join(line.rstrip() for line in raw.splitlines())
    assert norm == expected


def test_docx_reading_s1(reader):
    text = reader.extract_all_text(os.path.join(path, "sample.docx"))
    expected = "Hi, this is a docx in freeai-utils to check its function.\nThis second line is used to check the functions that able to separate lines."

    assert text == expected
    
    text = reader.extract_ordered_text(os.path.join(path, "sample.docx"))
    expected = "Hi, this is a docx in freeai-utils to check its function.\nThis second line is used to check the functions that able to separate lines."

    assert text == expected

def test_pdf_reading_s2(reader):
    raw = reader.extract_all_text(os.path.join(path, "sample2.pdf"))
    expected = "Test sample 2:\nFirst column Second column\nTask number 1: Check\n"
    norm = "\n".join(line.rstrip() for line in raw.splitlines())
    assert norm == expected
    
    raw = reader.extract_ordered_text(os.path.join(path, "sample2.pdf"))
    expected = "Test sample 2:\nFirst column Second column\nTask number 1: Check"
    norm = "\n".join(line.rstrip() for line in raw.splitlines())
    assert norm == expected


def test_docx_reading_s2(reader):
    text = reader.extract_all_text(os.path.join(path, "sample2.docx"))
    expected = "Test sample 2:\nFirst column\nSecond column\nTask number 1:\nCheck"

    print(repr(text))
    print(repr(expected))
    assert text == expected
    
    text = reader.extract_ordered_text(os.path.join(path, "sample2.docx"))
    expected = "Test sample 2:\nFirst column\tSecond column\nTask number 1:\tCheck\n"
    print(repr(text))
    print(repr(expected))
    assert text == expected
    
def test_pdf_reading_s3(reader):
    raw = reader.extract_all_text(os.path.join(path, "sample3.pdf"))
    expected = "Paragraph 1: This is the introduction.\nHeader 1 Header 2\nRow 1, Col 1 Row 1, Col 2\nParagraph 2: Following the first table.\nR1C1 R1C2 R1C3\nR2C1 R2C2 R2C3\nR3C1 R3C2 R3C3\nParagraph 3: Conclusion under the second table."
    norm = "\n".join(line.rstrip() for line in raw.splitlines())
    print(repr(norm))
    print(repr(expected))
    assert norm == expected
    
    raw = reader.extract_ordered_text(os.path.join(path, "sample3.pdf"))
    expected = "Paragraph 1: This is the introduction.\nHeader 1 Header 2\nRow 1, Col 1 Row 1, Col 2\nParagraph 2: Following the first table.\nR1C1 R1C2 R1C3\nR2C1 R2C2 R2C3\nR3C1 R3C2 R3C3\nParagraph 3: Conclusion under the second table."
    norm = "\n".join(line.rstrip() for line in raw.splitlines())
    print(repr(norm))
    print(repr(expected))
    assert norm == expected


def test_docx_reading_s3(reader):
    text = reader.extract_all_text(os.path.join(path, "sample3.docx"))
    expected = "Paragraph 1: This is the introduction.\nParagraph 2: Following the first table.\nParagraph 3: Conclusion under the second table.\nHeader 1\nHeader 2\nRow 1, Col 1\nRow 1, Col 2\nR1C1\nR1C2\nR1C3\nR2C1\nR2C2\nR2C3\nR3C1\nR3C2\nR3C3"
    print(repr(text))
    print(repr(expected))
    assert text == expected
    
    text = reader.extract_ordered_text(os.path.join(path, "sample3.docx"))
    expected = "Paragraph 1: This is the introduction.\nHeader 1\tHeader 2\nRow 1, Col 1\tRow 1, Col 2\nParagraph 2: Following the first table.\nR1C1\tR1C2\tR1C3\nR2C1\tR2C2\tR2C3\nR3C1\tR3C2\tR3C3\nParagraph 3: Conclusion under the second table."
    print(repr(text))
    print(repr(expected))
    assert text == expected