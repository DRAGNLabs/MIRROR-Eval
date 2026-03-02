import mirroreval

print(mirroreval.__version__)

config_path = "./settings.toml"

mirroreval.evaluate(config_path)
