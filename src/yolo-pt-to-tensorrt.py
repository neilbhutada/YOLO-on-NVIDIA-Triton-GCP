from pathlib import Path
from ultralytics import YOLO
model = YOLO('<yolo-pytorch-model>')
model.export(format="engine", dynamic=True, half = True, batch = 16)

