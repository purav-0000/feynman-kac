import typing
from typing import get_type_hints, get_origin, get_args


def apply_overrides(cfg: typing.Any, overrides: typing.List[str]):
    """
    Applies overrides to a config object using its type hints for casting.
    """
    # Get all type hints from the configuration class once.
    hints = get_type_hints(type(cfg))

    for kv in overrides or []:
        key, val_str = kv.split("=", 1)

        if key not in hints:
            raise KeyError(f"Error: '{key}' is not a valid configuration field. Check for typos.")

        # Get the specific type hint for the key being overridden.
        target_type = hints.get(key)
        processed_val = None

        # Handle 'None' value
        if val_str.lower() in ("none", "null"):
            processed_val = None

        # Handle boolean type
        elif target_type is bool:
            processed_val = val_str.lower() in ("true", "1")

        # Handle list types (e.g., List[int], List[float])
        elif get_origin(target_type) in (list, typing.List):
            # Get the list's inner type (e.g., int from List[int])
            element_type = get_args(target_type)[0] if get_args(target_type) else str
            # Create the list by casting each comma-separated value
            processed_val = [element_type(v.strip()) for v in val_str.split(',')]

        elif get_origin(target_type) is typing.Union:
            # If there's a comma, treat it as a list. Otherwise, a scalar.
            if ',' in val_str:
                # Find the list type within the Union, e.g., List[float]
                list_type = next((t for t in get_args(target_type) if get_origin(t) is list), None)
                if list_type:
                    # Get the list's element type (e.g., float) and cast each part
                    element_type = get_args(list_type)[0]
                    processed_val = [element_type(v.strip()) for v in val_str.split(',')]
            else:
                # Find the scalar type within the Union
                scalar_type = next((t for t in get_args(target_type) if get_origin(t) is not list), None)
                if scalar_type:
                    # Cast the whole string to the scalar type
                    processed_val = scalar_type(val_str)

        # Handle other simple types (int, float, str) if a hint exists
        elif target_type:
            processed_val = target_type(val_str)

        # Fallback for untyped fields (or if casting fails)
        else:
            processed_val = val_str

        setattr(cfg, key, processed_val)
