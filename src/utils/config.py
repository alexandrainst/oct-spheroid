import os

import torch
import yaml

# Mapping of YAML tags to libraries
LIBRARY_MAPPING = {"torch": torch}


class MultiLibDumper(yaml.SafeDumper):
    """Custom YAML dumper to handle torch, numpy, and tensorflow objects."""

    def represent_library_type(self, obj):
        """Convert special types into YAML-friendly representations."""
        for lib_name, lib in LIBRARY_MAPPING.items():
            for attr in dir(lib):
                if getattr(lib, attr) is obj:
                    return self.represent_scalar(f"!{lib_name}", attr)

        # Default to standard representation if no match found
        return self.represent_scalar("!unknown", str(obj))


# Register representers for each special type
for lib in LIBRARY_MAPPING.values():
    for dtype in dir(lib):
        try:
            obj = getattr(lib, dtype)
            if isinstance(obj, (torch.dtype, torch.memory_format)):
                MultiLibDumper.add_representer(type(obj), MultiLibDumper.represent_library_type)
        except AttributeError:
            pass  # Ignore attributes that cannot be accessed


class MultiLibLoader(yaml.SafeLoader):
    """Custom YAML loader to handle multiple library-specific types."""

    def construct_library_type(self, node):
        """Dynamically resolve types from specified libraries."""
        tag = node.tag.lstrip("!")  # Extract the library name (e.g., 'torch', 'numpy', 'tf')
        value = self.construct_scalar(node)  # Extract the scalar value

        if tag in LIBRARY_MAPPING:
            lib = LIBRARY_MAPPING[tag]
            return getattr(lib, value, value)  # Convert value to actual type if valid

        return value  # Return as is if library not found


for lib_tag in LIBRARY_MAPPING:
    MultiLibLoader.add_constructor(f"!{lib_tag}", MultiLibLoader.construct_library_type)


def load_yaml(file_path):
    """
    Load a YAML file and return its contents as a dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed YAML contents.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with open(file_path, "r") as file:
        return yaml.load(file, Loader=MultiLibLoader) or {}


def merge_configs(base_config, override_config):
    """
    Recursively merge two dictionaries, giving precedence to override_config.

    Args:
        base_config (dict): The base configuration dictionary.
        override_config (dict): The overriding configuration dictionary.

    Returns:
        dict: Merged configuration dictionary.
    """
    if not isinstance(base_config, dict) or not isinstance(override_config, dict):
        return override_config  # If it's not a dict, override completely

    merged = base_config.copy()
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def config_handler(overwrite_config_path: str, base_config_path: str = "./config/config.yaml") -> dict:
    """
    Load the base configuration file and merge it with the override configuration file.

    Args:
        overwrite_config_path (str): Path to the override configuration file.
        base_config_path (str): Path to the base configuration file. Defaults to "./config/config.yaml".

    Returns:
        dict: Merged configuration dictionary.
    """
    base_config = load_yaml(base_config_path)
    override_config = load_yaml(overwrite_config_path)
    return merge_configs(base_config, override_config)
