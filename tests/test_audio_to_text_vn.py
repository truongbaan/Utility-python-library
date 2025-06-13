import pytest
import gc
from freeai_utils.audio_to_text_vn import VN_Whisper

@pytest.fixture(scope="module")
def vn_model():
    model = VN_Whisper(device="gpu")
    yield model
    del model
    gc.collect()

def test_model_initialized(vn_model):
    assert vn_model.model is not None
    assert vn_model.device in ("cpu", "cuda")
    assert vn_model._device is not None
    assert vn_model._processor is not None
    with pytest.raises(AttributeError, match="Cannot reassign '_model'"):
        vn_model._model = None
    with pytest.raises(AttributeError, match="Cannot reassign '_device'"):
        vn_model._device = "cpu"
    with pytest.raises(AttributeError, match="Cannot reassign '_processor'"):
        vn_model._processor = "hi"

