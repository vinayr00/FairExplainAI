"""Logger utility setup."""
import logging
import sys
from pathlib import Path

def setup_logger(name: str, log_file: str, level=logging.INFO):
    """Sets up a logger with output to stdout and a file."""
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    # Ensure logs parent dir exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger
