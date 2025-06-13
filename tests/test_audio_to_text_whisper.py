import pytest
import gc
from freeai_utils.audio_to_text_whisper import OpenAIWhisper

@pytest.fixture(scope="module")
def whisper_model():
    model = OpenAIWhisper(model="tiny", device="gpu")  # lightweight model
    yield model
    del model
    gc.collect()

def test_model_initialized(whisper_model):
    assert whisper_model.model is not None
    assert whisper_model.device in ("cpu", "cuda")
    assert whisper_model.sample_rate == 16000
    with pytest.raises(AttributeError, match="Cannot reassign '_model'"):
        whisper_model._model = None
    with pytest.raises(AttributeError, match="Cannot reassign '_device'"):
        whisper_model._device = "cpu"

