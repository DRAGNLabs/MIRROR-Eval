import mirroreval

print(mirroreval.__version__)

config_path = "./settings.isolated_turn.toml"

mirroreval.evaluate(config_path)
