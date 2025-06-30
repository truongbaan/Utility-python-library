import pytest
import gc
from freeai_utils.localLLM import LocalLLM

@pytest.fixture(scope="module")
def lc_model():
    model = LocalLLM()
    yield model
    del model
    gc.collect()
    
def test_init(lc_model):
    assert lc_model.model is not None
    assert lc_model.tokenizer is not None
    assert lc_model.device in ("cpu", "cuda")
    assert lc_model._history == []
    assert lc_model._max_length == 4
    
def test_ask(lc_model):
    l_result = lc_model.ask([{"role": "user", "content": "This is a test of connecting. Please just answer \'yes\'"}])[0]
    assert l_result == "yes"
    s_result = lc_model.ask("This is a test of connecting. Please just answer \'yes\'")[0]
    assert s_result == "yes"
    
def test_ask_memories(lc_model):
    lc_model.ask_with_memories("Hi my name is An, please remember my name.")
    result = lc_model.ask_with_memories("This is a test of memory, please just tell me the provided name.")
    assert "An" in result[0]