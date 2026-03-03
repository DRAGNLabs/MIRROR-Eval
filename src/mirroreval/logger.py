import logging

# Create a global logger
logger = logging.getLogger("mirroreval")
logger.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

# Add handlers (avoid adding duplicates)
if not logger.handlers:
    logger.addHandler(ch)
