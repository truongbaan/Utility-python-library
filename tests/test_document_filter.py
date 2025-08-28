import pytest
import gc
from freeai_utils.document_filter import AIDocumentSearcher, DocumentFilter
from haystack import Document
@pytest.fixture(scope="module")
def searcher_model():
    model = AIDocumentSearcher(auto_init=False)
    yield model
    del model
    gc.collect()
    
@pytest.fixture(scope="module")
def filter_model():
    model = DocumentFilter(auto_init=False)
    yield model
    del model
    gc.collect()
    
def test_model_initialized(searcher_model):
    assert searcher_model.reader is not None
    assert searcher_model.threshold == 0.4
    assert searcher_model.max_per_doc == 2
    assert searcher_model.top_k == 4
    assert searcher_model.documents == []

    with pytest.raises(AttributeError, match="Cannot reassign 'reader' after initialization"):
        searcher_model.reader = None
    with pytest.raises(AttributeError, match="Cannot reassign '_reader' after initialization"):
        searcher_model._reader = None
        
def test_search_document(searcher_model):
    searcher_model._documents.append(Document(id = "0110", content="Python is a popular programming language known for its readability."))
    result = searcher_model.search_document("What is python?")

    assert result[0][1] >= 0.6
    assert len(result) == 1

def test_filter_initialized(filter_model):
    assert filter_model.documents == {}

def test_filter_document(filter_model):
    filter_model._documents["filePath"] = "Text 1. Text 2"
    result = filter_model.search_keyword("Text")
    sentences = result["filePath"]
    assert sentences[0] == "Text 1"
    assert sentences[1] == "Text 2"