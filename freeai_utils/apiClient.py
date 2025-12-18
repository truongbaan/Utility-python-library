import requests
from typing import Dict, Any, Optional, Tuple
from .utils import enforce_type
import logging
from .log_set_up import setup_logging

class ApiClient:
    _base_url: str
    _authToken: Optional[str]
    _timeout: int
    _cookiesPayload: Optional[Dict[str, Any]]
    logger: logging.Logger
    
    def __init__(self, base_url: str = "", authToken: Optional[str] = None, timeout: int = 30,
                 cookiesPayload : Optional[Dict[str, Any]] = None):
        
        enforce_type(base_url, str, "base_url")
        enforce_type(authToken, (str, type(None)), "authToken")
        enforce_type(timeout, int, "timeout")
        enforce_type(cookiesPayload, (Dict, type(None)), "cookiesPayload")
        
        super().__setattr__("_initialized", False)
        self._base_url = base_url.rstrip("/") #lock attribute base_url
        super().__setattr__("_initialized", True)
        self.logger = setup_logging(self.__class__.__name__)
        self._timeout = timeout
        self._authToken = authToken
        self._cookies = cookiesPayload if cookiesPayload else {}

    @property
    def base_url(self):
        return self._base_url
        
    @property
    def timeout(self):
        return self._timeout
    
    @timeout.setter
    def timeout(self, timeout : int):
        enforce_type(timeout, int, "timeout")
        self._timeout = timeout
        
    @property
    def authToken(self):
        """
        Get the current authentication token.

        Returns:
            Optional[str]: Bearer token if set, otherwise None.
        """
        return self._authToken
    
    @authToken.setter
    def authToken(self, authToken : str):
        enforce_type(authToken, str, "authToken")
        self._authToken = authToken
        
    @property
    def cookies(self):
        """
        Get the current cookies payload.

        Returns:
            Dict[str, Any]: Dictionary of cookies sent with each request.
        """
        return self._cookies
    
    @cookies.setter
    def cookies(self, cookies : Dict[str, Any]):
        enforce_type(cookies, Dict[str, Any], "cookies")
        self._cookies = cookies
    
    def changeBaseUrl(self, new_url : str):
        """
        Change the base URL of the API client after initialization.

        This method temporarily unlocks the internal base URL attribute,
        updates it, and then locks it again.

        Args:
            new_url (str):
                New base URL for the API client.

        Raises:
            TypeError:
                If new_url is not a string.
        """
        enforce_type(new_url, str, "new_url")
        super().__setattr__("_initialized", False)
        self._base_url = new_url.rstrip("/") #lock attribute base_url
        super().__setattr__("_initialized", True)
    
    def request(self, method: str = None, endpoint: str = "", authReq : bool = False, json_payload: Optional[Dict[str, Any]] = None,
        params_payload: Optional[Dict[str, Any]] = None, json_body_in_get: bool = False
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Send an HTTP request to the configured API endpoint.

        Args:
            method (str):
                HTTP method to use. Supported values:
                'get', 'post', 'put', 'patch', 'delete'.

            endpoint (str):
                API endpoint path (e.g. "/users/1").
                This is appended to the base URL.

            authReq (bool):
                Whether the request requires authentication.
                If True, a Bearer token must be set.

            json_payload (Optional[Dict[str, Any]]):
                JSON body to send with the request.
                Automatically serialized by the requests library.

            params_payload (Optional[Dict[str, Any]]):
                Query parameters appended to the URL.

            json_body_in_get (bool):
                Whether to allow a JSON body in GET requests.
                If False, JSON payloads are removed for GET requests.

        Returns:
            Tuple[int, Dict[str, Any]]:
                A tuple containing:
                - HTTP status code
                - Parsed JSON response as a dictionary, or an empty dict if no content

        Raises:
            TypeError:
                If input arguments are of invalid types.

            ValueError:
                If an unsupported HTTP method is provided.

            RuntimeError:
                If authentication is required but no token is set.

            requests.exceptions.RequestException:
                If a network or connection error occurs.
        """
        enforce_type(method, str, "method")
        enforce_type(authReq, bool, "authReq")
        enforce_type(json_body_in_get, bool, "json_body_in_get")
        
        method = method.lower() 
        if method not in {'get', 'post', 'put', 'patch', 'delete'}:
            raise ValueError("Invalid HTTP method")

        url = f"{self.base_url}{endpoint}"
        headers = {'accept': '*/*'}

        if authReq:
            if not self.authToken:
                raise RuntimeError("Auth requested but no token set")
            headers['Authorization'] = f'Bearer {self.authToken}'
            
        kwargs = {
            'headers': headers,
            'params': params_payload,
            'json': json_payload,
            'cookies': self._cookies,
            'timeout': self.timeout
        }

        if method == 'get' and not json_body_in_get:
            kwargs.pop('json', None)

        try:
            response = requests.request(method, url, **kwargs)
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Connection error: {e}")
            raise

        if not response.content:
            return response.status_code, {}

        try:
            return response.status_code, response.json()
        except ValueError:
            self.logger.error("Fail to parse to Json, continue to use raw text reponse")
            return response.status_code, {
                'warning': 'Response was not valid JSON',
                'raw_text': response.text
            }
            
    def __setattr__(self, name, value):
        # once initialized, prevent changing core internals
        if getattr(self, "_initialized", False) and name in ("_base_url", "_authToken", "_timeout", "_cookiesPayload"):
            raise AttributeError(f"Cannot reassign '{name}' after initialization")
        super().__setattr__(name, value)
        