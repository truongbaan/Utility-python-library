import pytest
import gc
from freeai_utils.document_filter import DocumentFilter
from haystack import Document
@pytest.fixture(scope="module")
def document_model():
    model = DocumentFilter(auto_init=False)
    yield model
    del model
    gc.collect()
    
def test_model_initialized(document_model):
    assert document_model.reader is not None
    assert document_model.threshold == 0.4
    assert document_model.max_per_doc == 2
    assert document_model.top_k == 4
    assert document_model.documents == []

    with pytest.raises(AttributeError, match="Cannot reassign 'reader' after initialization"):
        document_model.reader = None
    with pytest.raises(AttributeError, match="Cannot reassign '_reader' after initialization"):
        document_model._reader = None
        
def test_search_document(document_model):
    document_model._documents.append(Document(id = "0110", content="Python is a popular programming language known for its readability."))
    result = document_model.search_document("What is python?")

    assert result[0][1] >= 0.6
    assert len(result) == 1