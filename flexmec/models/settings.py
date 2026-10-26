import os

def _get_fail_log_path():
    default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs", "fail_sample_shape.txt")
    return os.environ.get("FLEXMEC_FAIL_LOG", default_path)


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_float(name, default, min_value=None, max_value=None):
    value = float(os.environ.get(name, str(default)))
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {value}")
    return value


def _env_int(name, default, min_value=None, max_value=None):
    value = int(os.environ.get(name, str(default)))
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {value}")
    return value
