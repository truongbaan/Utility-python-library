import pytest
from freeai_utils.apiClient import ApiClient
import gc

BASE_URL_JSON = "https://jsonplaceholder.typicode.com"

@pytest.fixture(scope="module")
def client():
    client = ApiClient(BASE_URL_JSON)
    yield client
    del client
    gc.collect()


def test_api_client_invalid_method_raises_value_error(client):
    """Tests if calling with an unsupported HTTP method raises ValueError."""
    with pytest.raises(ValueError, match="Invalid HTTP method"):
        client.request(method='head')
        
def test_api_client_non_string_method_raises_error(client):
    """Tests if calling with a non-string method raises an error."""
    with pytest.raises(TypeError):
        client.request(method=123)

def test_api_client_simple_get_success(client):
    """Tests a basic GET request with expected 200 status and valid JSON."""
    status, data = client.request(method='get', endpoint='/posts/1')

    assert status == 200
    assert isinstance(data, dict)
    assert data['id'] == 1
    assert data['userId'] == 1

def test_api_client_patch_with_json_success(client):
    """Tests a PATCH request with JSON payload for partial update."""
    payload = {"title": "new patched title"}

    status, data = client.request(
        method='patch',
        endpoint='/posts/1',
        json_payload=payload
    )

    assert status == 200
    assert isinstance(data, dict)
    assert data['id'] == 1
    assert data['title'] == "new patched title"
    assert 'body' in data

def test_api_client_post_with_json_success(client):
    """Tests a POST request with JSON payload."""
    payload = {"title": "foo", "body": "bar", "userId": 1}

    status, data = client.request(
        method='post',
        endpoint='/posts',
        json_payload=payload
    )

    assert status == 201
    assert isinstance(data, dict)
    assert data['title'] == "foo"
    assert 'id' in data

def test_api_client_delete_success(client):
    """Tests a DELETE request."""
    status, data = client.request(
        method='delete',
        endpoint='/posts/1'
    )

    assert status == 200
    assert data == {}
