"""Model registry — name → file path. Used by ac_base.load_class_from_path."""
from pathlib import Path

from utils.common import list_class_names

cur_path = Path(__file__).resolve().parent
model_name_to_path = list_class_names(cur_path)
