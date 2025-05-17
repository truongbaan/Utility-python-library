# config for pytest
import sys
import os

def pytest_configure():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'freeai_utils')))