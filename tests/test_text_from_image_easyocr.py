import pytest
import os
from freeai_utils.text_from_image_easyocr import Text_Extractor_EasyOCR 

path = os.path.dirname(os.path.abspath(__file__)) + "\\sample"

@pytest.fixture
def reader():
    return Text_Extractor_EasyOCR()

def test_ocr(reader):
    text = reader.get_text_from_image(os.path.join(path, "sample.png"))
    excepted = "Hi, This is a test for OCR_ Small text Big Text"
   
    assert text == excepted
    
    text = reader.get_text_from_image(os.path.join(path, "sample.jpg"))
    excepted = "Hi, This is a test for OCR_ Small text Big Text"
   
    assert text == excepted