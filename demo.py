import mirroreval

print(mirroreval.__version__)

config_path = "./.local/settings.toml"

mirroreval.evaluate(config_path)
