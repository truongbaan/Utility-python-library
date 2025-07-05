import pytest
import gc
from freeai_utils.language_detection import LangTranslator, LocalTranslator, M2M100Translator, MBartTranslator

def test_langtrans():
    model = LangTranslator()
    assert model.local_status == "backup"
    assert model.local_translator is not None
    assert model.translate(text_to_translate="Xin chào bạn", tgt_lang='en') == "Hello"
    assert model.detect_language("Xin chào bạn.")[0] == "vi"
    del model
    gc.collect()

def test_langtrans_local():
    model = LangTranslator(local_status="inactive")
    assert model.local_status is None
    assert model.local_translator is None
    assert model.translate(text_to_translate="Xin chào bạn", tgt_lang='en') == "Hello"
    assert model.detect_language("Xin chào bạn.")[0] == "vi"
    del model
    gc.collect()

def test_langtrans_fail():
    with pytest.raises(ValueError, match="local_status could only be \'active\', \'inactive\', \'backup\'. Current value: unknown"):
        model = LangTranslator(local_status="unknown")

def test_localtrans():
    model = LocalTranslator()
    assert model.model is not None
    assert model.translate(prompt="Xin chào bạn", tgt_lang='en') == "Hello friends."
    assert model.detect_language("Xin chào bạn") == "vi"
    del model
    gc.collect()

def test_localtrans_fail():
    with pytest.raises(ValueError):
        model = LocalTranslator(local_model_num=3)

def test_m2():
    model = M2M100Translator()
    assert model.model is not None
    assert model.tokenizer is not None
    assert model.device in ("cpu", "cuda")
    assert model.translate(text="Xin chào bạn", tgt_lang='en') == "Hello friends."
    assert model.detect_language("Xin chào bạn.") == "vi"
    del model
    gc.collect()

def test_mbart():
    model = MBartTranslator()
    assert model.model is not None
    assert model.tokenizer is not None
    assert model.device in ("cpu", "cuda")
    assert model.translate(text="Xin chào bạn", tgt_lang='en') == "Hi there."
    assert model.detect_language("Xin chào bạn.") == "vi"
    del model
    gc.collect()