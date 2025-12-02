#!/usr/bin/env python3
import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for inference configuration."""
    parser = argparse.ArgumentParser(description='Run inference with trained YOLOv8 models')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output images')
    parser.add_argument('--person_det_model', type=str, required=True, help='Path to person detection model weights')
    parser.add_argument('--ppe_detection_model', type=str, required=True, help='Path to PPE detection model weights')
    parser.add_argument('--conf_threshold', type=float, default=0.25, help='Confidence threshold for detections')
    parser.add_argument('--iou_threshold', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--img_size', type=int, default=640, help='Image size for inference')
    return parser.parse_args()


def load_classes(model_path: str) -> List[str]:
    """
    Load class names from the model's YAML configuration file.
    
    Falls back to default class names based on model type if YAML is not found.
    
    Args:
        model_path: Path to the model weights file
        
    Returns:
        List of class names for the model
    """
    model_dir = Path(model_path).parent.parent
    yaml_path = list(model_dir.glob('*.yaml'))
    
    if yaml_path:
        import yaml
        with open(yaml_path[0], 'r') as f:
            data = yaml.safe_load(f)
            if 'names' in data and data['names']:
                return data['names']
    
    # Determine class names based on model type
    if 'person_detection' in str(model_path):
        return ['person']
    elif 'ppe_detection' in str(model_path):
        return ['hard-hat', 'gloves', 'mask', 'glasses', 'boots', 'vest', 'ppe-suit', 'ear-protector', 'safety-harness']
    
    return []


def draw_detections(image: np.ndarray, detections: List, class_names: List[str], 
                   color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw bounding boxes and labels on the image.
    
    Args:
        image: Input image as numpy array
        detections: List of detection results [x1, y1, x2, y2, conf, cls_id]
        class_names: List of class names corresponding to class IDs
        color: BGR color tuple for drawing boxes and labels
        
    Returns:
        Image with drawn detections
    """
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        conf = float(det[4])
        cls_id = int(det[5])
        
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label = f"{cls_name} {conf:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - label_height - 5), (x1 + label_width, y1), color, -1)
        
        # Draw label text
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    return image

def process_image(image_path: Path, person_model: YOLO, ppe_model: YOLO, 
                  person_classes: List[str], ppe_classes: List[str], 
                  conf_threshold: float, iou_threshold: float, img_size: int) -> Optional[np.ndarray]:
    """
    Process a single image using two-stage detection pipeline.
    
    First detects persons in the image, then runs PPE detection on each cropped person region.
    
    Args:
        image_path: Path to the input image
        person_model: Loaded YOLO model for person detection
        ppe_model: Loaded YOLO model for PPE detection
        person_classes: List of person class names
        ppe_classes: List of PPE class names
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        img_size: Image size for model inference
        
    Returns:
        Image with drawn detections, or None if image could not be read
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # Run person detection on full image
    person_results = person_model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=img_size,
        verbose=False
    )[0]
    
    # Extract person detections
    person_detections = []
    for det in person_results.boxes.data.cpu().numpy():
        class_idx = int(det[5])
        if (len(person_classes) > 0 and class_idx < len(person_classes) and 
            person_classes[class_idx] == 'person') or class_idx == 0:
            person_detections.append(det)
    
    # Create output image with person detections
    output_image = image.copy()
    draw_detections(output_image, person_detections, person_classes, color=(0, 255, 0))
    
    # Process each detected person for PPE items
    all_ppe_detections = []
    for person_det in person_detections:
        x1, y1, x2, y2 = map(int, person_det[:4])
        
        # Clip coordinates to image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        
        # Skip invalid or tiny boxes
        if x2 - x1 < 10 or y2 - y1 < 10:
            continue
        
        # Crop person region from original image
        person_crop = image[y1:y2, x1:x2]
        
        # Run PPE detection on cropped region
        ppe_results = ppe_model.predict(
            source=person_crop,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=img_size,
            verbose=False
        )[0]
        
        # Transform PPE detections back to original image coordinates
        for det in ppe_results.boxes.data.cpu().numpy():
            crop_x1, crop_y1, crop_x2, crop_y2 = map(float, det[:4])
            conf = float(det[4])
            cls_id = int(det[5])
            
            # Convert coordinates to original image space
            orig_x1 = x1 + crop_x1
            orig_y1 = y1 + crop_y1
            orig_x2 = x1 + crop_x2
            orig_y2 = y1 + crop_y2
            
            all_ppe_detections.append([orig_x1, orig_y1, orig_x2, orig_y2, conf, cls_id])
    
    # Draw PPE detections in red
    draw_detections(output_image, all_ppe_detections, ppe_classes, color=(0, 0, 255))
    
    return output_image

def main() -> None:
    """Main execution function for running inference on images."""
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading person detection model...")
    person_model = YOLO(args.person_det_model)
    
    print("Loading PPE detection model...")
    ppe_model = YOLO(args.ppe_detection_model)
    
    person_classes = load_classes(args.person_det_model)
    ppe_classes = load_classes(args.ppe_detection_model)
    
    print(f"Person classes: {person_classes}")
    print(f"PPE classes: {ppe_classes}")
    
    # Collect all image files from input directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(list(Path(args.input_dir).glob(f"*{ext}")))
    
    print(f"Found {len(image_paths)} images in {args.input_dir}")
    
    # Process each image through the detection pipeline
    for image_path in image_paths:
        print(f"Processing {image_path.name}...")
        
        output_image = process_image(
            image_path,
            person_model,
            ppe_model,
            person_classes,
            ppe_classes,
            args.conf_threshold,
            args.iou_threshold,
            args.img_size
        )
        
        if output_image is not None:
            output_path = Path(args.output_dir) / image_path.name
            cv2.imwrite(str(output_path), output_image)
            print(f"Saved output to {output_path}")
    
    print("Inference complete!")


if __name__ == "__main__":
    main()
