import pytest
import gc
from freeai_utils.image_to_text import ImageCaptioner

@pytest.fixture(scope="module")
def img_cap():
    model = ImageCaptioner()
    yield model
    del model
    gc.collect()
    
def test_init(img_cap):
    assert img_cap.device in ("cpu", "cuda")
    assert img_cap.model is not None
    assert img_cap.processor is not None