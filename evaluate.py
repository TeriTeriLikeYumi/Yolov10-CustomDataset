import argparse
from pathlib import Path
from ultralytics import YOLO
import model_config

def evaluate_model(model_path, data_path):
    # Load the model
    model = YOLO(model_path)
    
    # Validate the model
    metrics = model.val(data=data_path)
    
    # Print the evaluation metrics
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv10 model")
    parser.add_argument(
        "--model_type", 
        type=str, 
        choices=["Yolov10s Detection", "Fruits Detection"], 
        required=True, 
        help="Type of model to evaluate"
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True, 
        help="Path to the dataset configuration file (data.yaml)"
    )
    
    args = parser.parse_args()
    
    if args.model_type == "Yolov10s Detection":
        model_path = model_config.DETECTION_MODEL
    elif args.model_type == "Fruits Detection":
        model_path = model_config.CUSTOM_MODEL
    else:
        raise ValueError("Invalid model type selected")
    
    evaluate_model(model_path, args.data_path)