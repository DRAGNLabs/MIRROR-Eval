import logging

# Create a global logger
logger = logging.getLogger("mirroreval")
logger.setLevel(logging.INFO)  # Set minimum log level

# Formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

# File handler (optional)
fh = logging.FileHandler("app.log")
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

# Add handlers (avoid adding duplicates)
if not logger.handlers:
    logger.addHandler(ch)
    logger.addHandler(fh)
