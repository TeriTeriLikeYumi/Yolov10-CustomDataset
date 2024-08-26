# Yolov10 - Streamlit Demo
* An Object detection app created using streamlit & train by Google Colab
* Model used: https://github.com/THU-MIG/yolov10
* App link: https://yolov10-app.streamlit.app/
* Dataset: https://universe.roboflow.com/fruit-detection-w707e/fruit-detection-deqvb
# App feature
* Custom-trained yolov10 model **for detecting fruits** (5 classes: [Apples', 'Onions', 'Pineapple', 'Tomatoes', 'Watermelons'])
* Switch between pre-trained yolov10 model and custom one fine-tuned on fruit dataset
# YOlOv10 Pre-trained model performance
COCO

| Model | Test Size | #Params | FLOPs | AP<sup>val</sup> | Latency |
|:---------------|:----:|:---:|:--:|:--:|:--:|
| [YOLOv10-N](https://huggingface.co/jameslahm/yolov10n) |   640  |     2.3M    |   6.7G   |     38.5%     | 1.84ms |
| [YOLOv10-S](https://huggingface.co/jameslahm/yolov10s) |   640  |     7.2M    |   21.6G  |     46.3%     | 2.49ms |
| [YOLOv10-M](https://huggingface.co/jameslahm/yolov10m) |   640  |     15.4M   |   59.1G  |     51.1%     | 4.74ms |
| [YOLOv10-B](https://huggingface.co/jameslahm/yolov10b) |   640  |     19.1M   |  92.0G |     52.5%     | 5.74ms |
| [YOLOv10-L](https://huggingface.co/jameslahm/yolov10l) |   640  |     24.4M   |  120.3G   |     53.2%     | 7.28ms |
| [YOLOv10-X](https://huggingface.co/jameslahm/yolov10x) |   640  |     29.5M    |   160.4G   |     54.4%     | 10.70ms |
# Custom-trained model's result:
|    Custom-trained models    |      mAP50      | mAP50-95|
|---------------              |---------------  |-------  |
| Fruits detection (yolov10s) |       0.822     | 0.626   |

* mAP50 and mAP50-95 are metrics used to evaluate object detection models.
* mAP50 measures the average precision at an Intersection over Union (IoU) threshold of 0.5, while mAP50-95 considers the average precision across IoU thresholds from 0.5 to 0.95.
* Higher values indicate better accuracy and robustness in detecting objects across different IoU levels.
* IOU is the ratio of the area of overlap between the predicted and actual bounding boxes to the area of their union
# How to run the model
## Create virtual environment
```python
$ conda create -n <env_name> -y python=3.11
$ conda activate <env_name>
$pip3 install -r requirements.txt
```

## Streamlit app
```python
streamlit run app.py
```
## Training
```
from ultralytics import YOLO

# Load a model
model = YOLO('yolov10s.pt')  # load a pretrained model (recommended for training)

# Train the model
dataset_path = dataset.location + '/data.yaml'
model.train(data = dataset_path, epochs=100, batch=32,imgsz=640, plots=True)
```
## Validate
```
from ultralytics import YOLO
# Load the trained model
model = YOLO('/content/runs/detect/train/weights/best.pt')

# Validate the model
metrics = model.val()
print(metrics.box.map)  # mAP50-95
```