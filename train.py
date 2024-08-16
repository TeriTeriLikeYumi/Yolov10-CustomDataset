from ultralytics import YOLOv10

model = YOLOv10()
model = YOLOv10(yolo10n.pt)

model.train(data='coco.yaml', epochs=500, batch=4, imgsz=640)