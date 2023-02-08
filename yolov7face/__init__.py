import sys

sys.path.append("yolov7")

from .anonymize import FaceAnonymizer
from .face import YOLOv7Configs, YOLOv7Face
from .model import YOLOv7Model, DEFAULT_MODEL as YOLOV7_WIDERFACE_MODEL


__all__ = ['FaceAnonymizer', 'YOLOv7Configs', 'YOLOv7Face', 'YOLOv7Model', 'YOLOV7_WIDERFACE_MODEL']

__version__ = '0.0.11'
