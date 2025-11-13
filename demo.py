import mirroreval

print(mirroreval.__version__)

config_path = "./settings-local.toml"

mirroreval.evaluate(config_path)
