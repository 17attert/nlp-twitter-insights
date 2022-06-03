import pathlib
import os

# Get project's root directory
dir_path = os.path.dirname(os.path.realpath(__file__))
root_dir = pathlib.Path(dir_path).parent

# Utilities directories
UTILS_DIR = root_dir.joinpath("utilities")

# Model directories
MODELS_DIR = root_dir.joinpath("models")
MODELS_CACHE_DIR = MODELS_DIR.joinpath("cache")

# Data directories
DATA_DIR = root_dir.joinpath("jigsaw-toxic-comment-classification-challenge")
