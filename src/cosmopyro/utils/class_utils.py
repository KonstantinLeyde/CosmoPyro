from collections.abc import Mapping, Iterable
from typing import Any
import inspect

def get_init_vars(class_instance):
    # Get the argument names of __init__ (excluding 'self')
    init_params = inspect.signature(class_instance.__init__).parameters
    init_vars = {k: v for k, v in class_instance.__dict__.items() if k in init_params}

    # Include properties that are mentioned in __init__
    properties = {name: getattr(class_instance, name) for name in init_params
                   if isinstance(getattr(type(class_instance), name, None), property)}

    # Combine instance variables and properties
    all_vars = {**init_vars, **properties}

    return all_vars



def update_dict(keys: Iterable[str], value: Any, dictionary: Mapping[str, Any]) -> None:
    """Update a nested dictionary with a new value in place.
    
    If updated value is a dictionary, it will be merged with the existing dictionary
    such that the existing dictionary will be updated with the new values.

    Will create new dictionaries if the keys do not exist.
    """
    if not isinstance(dictionary, Mapping):
        raise TypeError(f"dictionary must be a dictionary: {dictionary}")
    # If only one key, update dictionary with value
    # Else, recursively update the dictionary
    if len(keys) == 1:
        if keys[0] in dictionary:
            current_val = dictionary[keys[0]]
            if isinstance(current_val, Mapping):
                # If the value is a dictionary, merge the two dictionaries
                # Else if the value is not a dictionary, raise an error
                if isinstance(value, Mapping):
                    # Only merge if there are no overlapping keys
                    overlap = set(current_val.keys()) & set(value.keys())
                    if overlap:
                        raise ValueError(f"Dictionaries have overlapping keys: {overlap}")
                    current_val.update(value)
                else:
                    raise ValueError(f"Cannot overwrite existing dictionary with value {value}")
            elif isinstance(value, Mapping):
                # If the key is in the dictionary but not a dictionary
                # and the new value is a dictionary, raise an error
                raise ValueError(f"Cannot overwrite existing value with dictionary: {current_val}")
            else:
                # If the key is in the dictionary but not a dictionary
                # and the new value is not a dictionary, overwrite the value
                dictionary[keys[0]] = value
        else:
            # If the key does not exist in the dictionary, create a new key-value pair
            dictionary[keys[0]] = value
    elif len(keys) == 0:
        raise ValueError("No keys provided")
    else:
        # If key does not exist, create a new dictionary
        if keys[0] not in dictionary:
            dictionary[keys[0]] = {}
        # If the value is not a dictionary, raise an error
        if not isinstance(dictionary[keys[0]], Mapping):
            raise TypeError(f"Cannot recurse into a non-dictionary value: {dictionary[keys[0]]}")
        # Recursively update the dictionary
        update_dict(keys[1:], value=value, dictionary=dictionary[keys[0]])