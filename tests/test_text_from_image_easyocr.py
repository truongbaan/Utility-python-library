import pytest
from freeai_utils.text_from_image_easyocr import Text_Extractor_EasyOCR 
import gc

@pytest.fixture(scope="module")
def ocr_extractor():
    extractor = Text_Extractor_EasyOCR(language=["en"], prefer_device="cpu")
    yield extractor
    del extractor
    gc.collect()

def test_capture_region_setter_getter(ocr_extractor):
    region = (100, 100, 500, 400)
    ocr_extractor.capture_region = region
    assert ocr_extractor.capture_region == region

def test_set_capture_region_percent(ocr_extractor):
    region = ocr_extractor.set_capture_region(crop_left=10, crop_right=10, crop_up=5, crop_down=5)
    assert isinstance(region, tuple)
    assert len(region) == 4
    
def test_cannot_modify_after_init(ocr_extractor):
    with pytest.raises(AttributeError, match="Cannot reassign '_reader'"):
        ocr_extractor._reader = "something"
    with pytest.raises(AttributeError, match="Cannot reassign '_screen_width'"):
        ocr_extractor._screen_width = 800
    with pytest.raises(AttributeError, match="Cannot reassign '_screen_height'"):
        ocr_extractor._screen_height = 600

def test_text_from_dummy_image(tmp_path, ocr_extractor):
    # Create a dummy image
    from PIL import Image, ImageDraw
    dummy_img_path = tmp_path / "dummy.png"
    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 40), "Test", fill=(0, 0, 0))
    img.save(dummy_img_path)

    result = ocr_extractor.get_text_from_image(str(dummy_img_path))
    assert isinstance(result, str)