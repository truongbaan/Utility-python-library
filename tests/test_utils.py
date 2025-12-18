import pytest
from freeai_utils.utils import enforce_type, apiRequest

BASE_URL_JSON = "https://jsonplaceholder.typicode.com"

# ====================================================================
#  TESTS FOR enforce_type 
# ====================================================================

def test_enforce_type_valid_single_type():
    """Tests if a single type check (str) passes correctly."""
    value = "hello"
    expected = str
    arg_name = "test_arg"
    
    # Assert that the function runs without raising an exception
    enforce_type(value, expected, arg_name) 

def test_enforce_type_valid_multiple_types():
    value = 3.14
    expected = (int, float, str)
    arg_name = "test_numeric"
    
    enforce_type(value, expected, arg_name)

def test_enforce_type_invalid_single_type():
    value = 100
    expected = str
    arg_name = "test_string"
    
    with pytest.raises(TypeError) as excinfo:
        enforce_type(value, expected, arg_name)
    
    # Check the error message format
    assert "Argument 'test_string' must be of type str" in str(excinfo.value)
    assert "received int" in str(excinfo.value)

def test_enforce_type_invalid_multiple_types():
    value = [1, 2] # This is a list
    expected = (int, float, str)
    arg_name = "test_scalar"
    
    with pytest.raises(TypeError) as excinfo:
        enforce_type(value, expected, arg_name)
        
    assert "Argument 'test_scalar' must be of type int, float, str" in str(excinfo.value)
    assert "received list" in str(excinfo.value)

def test_enforce_type_handles_none_without_optional():
    value = None
    expected = str
    arg_name = "test_non_optional"
    
    with pytest.raises(TypeError) as excinfo:
        enforce_type(value, expected, arg_name)
        
    # None is of type 'NoneType'
    assert "must be of type str" in str(excinfo.value)
    assert "received NoneType" in str(excinfo.value)

# ====================================================================
#  TESTS FOR apiRequest
# ====================================================================


def test_api_request_invalid_action_raises_value_error():
    """Tests if calling with an unsupported HTTP action raises ValueError."""
    with pytest.raises(ValueError, match="must be in these input"):
        apiRequest(method='head', endpoint=BASE_URL_JSON) # 'head' is unsupported

def test_api_request_non_string_action_raises_type_error():
    """Tests if calling with a non-string action raises TypeError."""
    with pytest.raises(TypeError, match="must be a string"):
        apiRequest(method=123, endpoint=BASE_URL_JSON)

def test_api_request_simple_get_success():
    """Tests a basic GET request with expected 200 status and valid JSON."""
    status, data = apiRequest(method='get', endpoint=f"{BASE_URL_JSON}/posts/1")
    assert status == 200
    assert isinstance(data, dict)
    assert data['id'] == 1
    assert data['userId'] == 1

def test_api_request_get_with_params_success():
    """Tests a GET request where params are correctly included."""
    params = {"userId": 1, "id": 5}
    status, data = apiRequest(method='GET', endpoint=f"{BASE_URL_JSON}/posts", params_payload=params)
    assert status == 200
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]['id'] == 5
    
def test_api_request_patch_with_json_success():
    """Tests a PATCH request with JSON payload for partial update."""
    payload = {"title": "new patched title"}
    status, data = apiRequest(method='patch', endpoint=f"{BASE_URL_JSON}/posts/1", json_payload=payload)
    assert status == 200 # Expect 200 OK for PATCH success
    assert isinstance(data, dict)
    assert data['id'] == 1
    assert data['title'] == "new patched title"
    assert 'body' in data

def test_api_request_post_with_json_success():
    """Tests a POST request with JSON payload."""
    payload = {"title": "foo", "body": "bar", "userId": 1}
    status, data = apiRequest(method='post', endpoint=f"{BASE_URL_JSON}/posts", json_payload=payload)
    assert status == 201 # Expect 201 Created for POST on this test API
    assert isinstance(data, dict)
    assert data['title'] == "foo"
    assert 'id' in data # Should have a generated ID

def test_api_request_delete_success():
    """Tests a DELETE request."""
    status, data = apiRequest(method='delete', endpoint=f"{BASE_URL_JSON}/posts/1")
    assert status == 200
    assert data == {} 