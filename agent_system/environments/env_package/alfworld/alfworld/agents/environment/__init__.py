"""TextWorld environment used by the released ALFWorld pipeline."""
def get_environment(env_type):
    if env_type != 'AlfredTWEnv':
        raise ValueError(f'Only AlfredTWEnv is included, got {env_type!r}')
    from .alfred_tw_env import AlfredTWEnv
    return AlfredTWEnv
