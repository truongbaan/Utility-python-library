import time
from functools import wraps
import shutil
import requests
from typing import Dict, Any, Tuple, Optional

def enforce_type(value, expected_types, arg_name):
    """Verify type of the value to ensure it is valid type before entering functions"""
    if not isinstance(value, expected_types):
        expected_names = [t.__name__ for t in expected_types] if isinstance(expected_types, tuple) else [expected_types.__name__]
        expected_str = ", ".join(expected_names)
        raise TypeError(f"Argument '{arg_name}' must be of type {expected_str}, but received {type(value).__name__}")

#  A decorator that measures and prints the execution time of a function.
def time_it(func):
    @wraps(func) 
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds.")
        return result
    return wrapper

def colorize(text: str, color_code: str) -> str:
    """Returns a string with the specified ANSI color and a reset code."""
    return f"\033[{color_code}m{text}\033[0m"

def get_free_space_gb(path='/') -> float:
    #Defaults to the root directory ('/') which typically represents the system drive.
    try:
        disk_stats = shutil.disk_usage(path)

        # Convert bytes to gigabytes
        free_space_bytes = disk_stats.free
        free_space_gb = free_space_bytes / (1024**3)
        return free_space_gb
    except Exception as e:
        print(f"Error checking disk space: {e}")
        return None

def apiRequest(method : str, endpoint : str, authorizedToken : str = None, json_payload: Optional[Dict[str, Any]] = None, 
               params_payload: Optional[Dict[str, Any]] = None, cookies_payload: Optional[Dict[str, Any]] = None,
               json_body_in_get : bool = False, timeout : int = 30
               ) -> Tuple[int, Dict[str, Any]]:
    """
    Makes a specified HTTP request (PUT, POST, DELETE, GET, PATCH) to an API endpoint.

    The response content is expected to be JSON. Authorization is added via
    a 'Bearer' token if provided.

    Args:
        method (str): The HTTP method to use. Must be one of 'put', 'post', 
                      'delete', 'patch', or 'get'.
        endpoint (str): The full URL of the API endpoint.
        authorizedToken (str, optional): The authorization token (e.g., a JWT). 
                                        If provided, it is included as 'Authorization: Bearer <token>' header. 
                                        Defaults to None.
        json_payload (Dict[str, Any], optional): The data to be sent in the request body (for PUT, POST, DELETE) or even GET if allows
                                            Defaults to None.
        params_payload (Dict[str, Any], optional): The data to be sent in the query parameters (for GET). 
                                            Defaults to None.
        cookies_payload (Dict[str, Any], optional): The data to be sent in as the cookies. 
                                            Defaults to None.
        json_body_in_get (bool): This allows the body to be sent in the get HTTP method. Defaults to False.
        timeout (int): The time it wait for the connection

    Returns: Tuple[int, Dict[str, Any]]: A tuple containing the HTTP status code and the JSON-decoded response body.

    Raises:
        ValueError: If the 'method' argument is not one of the allowed HTTP methods.
        TypeError: If the 'method' argument is not str.
        requests.exceptions.RequestException: For connection issues after printing a console message.
    """
    
    if authorizedToken is None:
        headers = {'accept': '*/*'}
    else: headers = {'accept': '*/*','Authorization': f'Bearer {authorizedToken}'}
    
    methodList = ['put', 'post', 'delete', 'get', 'patch']
    if not isinstance(method, str):
        raise TypeError(f"Argument 'method' must be a string, got {type(method).__name__}")
    method = method.lower()
    
    if method not in methodList:
        raise ValueError(f"Argument method must be in these input {','.join(methodList)}")
    try:
        req_kwargs = {
            'headers': headers,
            'params': params_payload,
            'json': json_payload, 
            'cookies': cookies_payload, 
            'timeout': timeout
        }

        if method == 'put':
            response = requests.put(endpoint, **req_kwargs)
        elif method == 'post':
            response = requests.post(endpoint, **req_kwargs)
        elif method == 'delete':
            response = requests.delete(endpoint, **req_kwargs)
        elif method == 'patch':
            response = requests.patch(endpoint, **req_kwargs)
        elif method == 'get':
            if not json_body_in_get:
                req_kwargs.pop('json', None)
            response = requests.get(endpoint, **req_kwargs) 
        
        if not response.content: 
            return response.status_code, {}
        try:
            return response.status_code, response.json()
        except requests.exceptions.JSONDecodeError:
            return response.status_code, {
                'warning': 'Response content was not valid JSON. Returning raw text.',
                'raw_text': response.text
                }

    except requests.exceptions.RequestException as e:
        print(f"A connection error occurred request: {e}")
        raise e

#how to use
if __name__== "__main__":
    @time_it
    def say_hello():
        print("Hello world!")  
    say_hello()