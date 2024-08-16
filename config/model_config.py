from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())


# Source
SOURCES_LIST = ["Image", "Video"]


# DL model config
DETECTION_MODEL_DIR = ROOT / 'weights' / 'detection'
YOLOv10n = DETECTION_MODEL_DIR / "yolov10n.pt"
YOLOv10s = DETECTION_MODEL_DIR / "yolov10s.pt"
YOLOv10m = DETECTION_MODEL_DIR / "yolov10m.pt"
YOLOv10l = DETECTION_MODEL_DIR / "yolov10l.pt"
YOLOv10x = DETECTION_MODEL_DIR / "yolov10x.pt"

DETECTION_MODEL_LIST = [
    "yolov10n.pt",
    "yolov10s.pt",
    "yolov10m.pt",
    "yolov10l.pt",
    "yolov10x.pt"]