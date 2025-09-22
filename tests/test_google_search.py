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
    tries = 0
    result = ""
    while "ha noi" not in result.lower() and "hanoi" not in result.lower() and tries < 3:
        result = webscrap.search("What is the capital of Vietnam?")
        tries += 1
    tries = 0
    result2 = ""
    while "ha noi" not in result2.lower() and "hanoi" not in result2.lower() and tries < 3:
        result2 = webscrap.quick_search("What is the capital of Vietnam?")
        tries += 1
    
    assert "ha noi" in result.lower() or "hanoi" in result.lower()
    assert "ha noi" in result2.lower() or "hanoi" in result2.lower()