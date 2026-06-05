from pathlib import Path

import yaml

# The config file is expected to be in the project root
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config():
    """Loads the configuration from the YAML file."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


# Load the configuration once
_config_data = load_config()


def get_config(section: str, key: str, default=None):
    """
    Retrieves a configuration value from a specific section and key.

    :param section: The top-level section in the YAML (e.g., 'process', 'train')
    :param key: The key within that section
    :param default: Default value if the key or section is not found
    :return: The configuration value or the default value
    """
    section_dict = _config_data.get(section, {})
    return section_dict.get(key, default)


# For easy access to the whole dict if needed
config = _config_data
