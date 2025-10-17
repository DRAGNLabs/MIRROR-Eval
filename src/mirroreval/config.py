from dynaconf import Dynaconf

# Create a settings object without specifying a file yet
settings = Dynaconf()


def init_settings(settings_file_path=None):
    """
    Optionally initialize the global `settings` object from a user-supplied TOML.
    This must be called before any other module accesses `settings` if a custom file is needed.
    """
    global settings
    if settings_file_path:
        settings.load_file(path=[settings_file_path])

    # In the settings object, set the path to the settings file for reference
    settings.slurm_job.settings_file_path = str(settings_file_path)
