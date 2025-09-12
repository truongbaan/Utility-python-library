def enforce_type(value, expected_types, arg_name):
    """Verify type of the value to ensure it is valid type before entering functions"""
    if not isinstance(value, expected_types):
        expected_names = [t.__name__ for t in expected_types] if isinstance(expected_types, tuple) else [expected_types.__name__]
        expected_str = ", ".join(expected_names)
        raise TypeError(f"Argument '{arg_name}' must be of type {expected_str}, but received {type(value).__name__}")
