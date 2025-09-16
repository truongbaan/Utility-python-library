import pytest
import gc
from freeai_utils.google_search import WebScraper

@pytest.fixture(scope="module")
def webscrap():
    model = WebScraper()
    yield model
    del model
    gc.collect()
    
def test_model_initialized(webscrap):
    assert webscrap.limit_word == 500
    assert webscrap.num_results == 5
    assert webscrap.user_agent == "Mozilla/5.0"

def test_search(webscrap):
    result = webscrap.search("What is the capital of Vietnam?")
    assert "ha noi" in result.lower() or "hanoi" in result.lower()
    result2 = webscrap.quick_search("What is the capital of Vietnam?")
    assert "ha noi" in result2.lower() or "hanoi" in result2.lower()