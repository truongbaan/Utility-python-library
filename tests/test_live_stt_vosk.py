import pytest
import gc
from freeai_utils.live_stt_vosk import STT_Vosk

@pytest.fixture(scope="module")
def vosk_model():
    model = STT_Vosk()
    yield model
    del model
    gc.collect()
    
def test_init(vosk_model):
    assert vosk_model._model is not None
    assert vosk_model._rec is not None
    assert vosk_model._sample_rate == 16000
    assert vosk_model._dtype == "int16"
    assert vosk_model._channels == 1
    assert vosk_model._frame_duration == 0.1
    assert vosk_model._blocksize == (int)((0.1) * (16000))