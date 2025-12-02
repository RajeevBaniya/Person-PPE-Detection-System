#!/usr/bin/env python3
import os
import argparse
import yaml
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for model training configuration."""
    parser = argparse.ArgumentParser(description='Train YOLOv8 models for person and PPE detection')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the YOLOv8 formatted dataset')
    parser.add_argument('--weights_dir', type=str, default='weights', help='Directory to save model weights')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=640, help='Image size for training')
    parser.add_argument('--person_model', type=str, default='yolov8n.pt', help='Base model for person detection')
    parser.add_argument('--ppe_model', type=str, default='yolov8n.pt', help='Base model for PPE detection')
    return parser.parse_args()


def create_person_dataset(data_dir: str, output_dir: str) -> str:
    """
    Create a dataset containing only person class annotations.
    
    Filters the full dataset to include only person detections (class 0).
    
    Args:
        data_dir: Source directory containing the full dataset
        output_dir: Destination directory for person-only dataset
        
    Returns:
        Path to the generated dataset.yaml file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # Copy all images
    for split in ['train', 'val', 'test']:
        src_img_dir = os.path.join(data_dir, split, 'images')
        dst_img_dir = os.path.join(output_dir, split, 'images')
        
        for img_file in os.listdir(src_img_dir):
            shutil.copy(os.path.join(src_img_dir, img_file), os.path.join(dst_img_dir, img_file))
    
    # Filter labels to keep only person class (class 0)
    for split in ['train', 'val', 'test']:
        src_label_dir = os.path.join(data_dir, split, 'labels')
        dst_label_dir = os.path.join(output_dir, split, 'labels')
        
        for label_file in os.listdir(src_label_dir):
            with open(os.path.join(src_label_dir, label_file), 'r') as f:
                lines = f.readlines()
            
            person_lines = [line for line in lines if line.strip().startswith('0 ')]
            
            with open(os.path.join(dst_label_dir, label_file), 'w') as f:
                f.writelines(person_lines)
    
    # Generate dataset.yaml configuration
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n\n")
        f.write("nc: 1\n")
        f.write("names: ['person']\n")
    
    return yaml_path

def create_ppe_dataset_with_cropping(data_dir: str, output_dir: str, classes_file: str) -> str:
    """
    Create a PPE detection dataset by cropping person regions from full images.
    
    This function processes images to crop person bounding boxes and maps PPE annotations
    to the cropped coordinate space. Each person detection becomes a separate training sample.
    
    Args:
        data_dir: Source directory containing the full dataset
        output_dir: Destination directory for cropped PPE dataset
        classes_file: Path to file containing class names
        
    Returns:
        Path to the generated dataset.yaml file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load class names from file
    with open(classes_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    
    # Create PPE-only class mapping (exclude person class)
    ppe_classes = [cls for cls in class_names if cls != 'person']
    original_class_map = {i: cls for i, cls in enumerate(class_names)}
    new_class_map = {cls: i for i, cls in enumerate(ppe_classes)}
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # Process each data split
    for split in ['train', 'val', 'test']:
        src_img_dir = os.path.join(data_dir, split, 'images')
        src_label_dir = os.path.join(data_dir, split, 'labels')
        dst_img_dir = os.path.join(output_dir, split, 'images')
        dst_label_dir = os.path.join(output_dir, split, 'labels')
        
        # Process each image and its annotations
        for img_file in os.listdir(src_img_dir):
            img_path = os.path.join(src_img_dir, img_file)
            label_file = os.path.join(src_label_dir, Path(img_file).stem + '.txt')
            
            if not os.path.exists(label_file):
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img_height, img_width = img.shape[:2]
            
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # Extract person bounding boxes from annotations
            person_boxes = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5 and parts[0] == '0':
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    # Convert from normalized YOLO format to pixel coordinates
                    x_center *= img_width
                    y_center *= img_height
                    width *= img_width
                    height *= img_height
                    
                    # Calculate bounding box corners
                    xmin = max(0, int(x_center - width / 2))
                    ymin = max(0, int(y_center - height / 2))
                    xmax = min(img_width, int(x_center + width / 2))
                    ymax = min(img_height, int(y_center + height / 2))
                    
                    person_boxes.append((xmin, ymin, xmax, ymax))
            
            # Crop each person detection and create individual training samples
            for i, (xmin, ymin, xmax, ymax) in enumerate(person_boxes):
                person_img = img[ymin:ymax, xmin:xmax]
                
                # Skip invalid or tiny crops
                if person_img.shape[0] < 10 or person_img.shape[1] < 10:
                    continue
                
                # Generate unique filename for cropped image
                new_img_file = f"{Path(img_file).stem}_person_{i}.jpg"
                new_img_path = os.path.join(dst_img_dir, new_img_file)
                
                cv2.imwrite(new_img_path, person_img)
                
                # Create corresponding label file
                new_label_file = f"{Path(img_file).stem}_person_{i}.txt"
                new_label_path = os.path.join(dst_label_dir, new_label_file)
                
                # Map PPE annotations to cropped coordinate space
                with open(new_label_path, 'w') as f_out:
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            
                            if class_id == 0:
                                continue
                            
                            # Verify class is a PPE item
                            class_name = original_class_map.get(class_id)
                            if class_name not in ppe_classes:
                                continue
                            
                            # Convert from normalized to pixel coordinates
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            x_center_px = x_center * img_width
                            y_center_px = y_center * img_height
                            width_px = width * img_width
                            height_px = height * img_height
                            
                            # Calculate bounding box corners
                            box_xmin = x_center_px - width_px / 2
                            box_ymin = y_center_px - height_px / 2
                            box_xmax = x_center_px + width_px / 2
                            box_ymax = y_center_px + height_px / 2
                            
                            # Check if PPE box overlaps with person box
                            if (box_xmin < xmax and box_xmax > xmin and 
                                box_ymin < ymax and box_ymax > ymin):
                                
                                # Calculate intersection region
                                inter_xmin = max(xmin, box_xmin)
                                inter_ymin = max(ymin, box_ymin)
                                inter_xmax = min(xmax, box_xmax)
                                inter_ymax = min(ymax, box_ymax)
                                
                                # Transform coordinates to cropped image space
                                new_x_center = (inter_xmin + inter_xmax) / 2 - xmin
                                new_y_center = (inter_ymin + inter_ymax) / 2 - ymin
                                new_width = inter_xmax - inter_xmin
                                new_height = inter_ymax - inter_ymin
                                
                                # Normalize to cropped image dimensions
                                person_width = xmax - xmin
                                person_height = ymax - ymin
                                new_x_center /= person_width
                                new_y_center /= person_height
                                new_width /= person_width
                                new_height /= person_height
                                
                                # Write annotation with remapped class ID
                                new_class_id = new_class_map[class_name]
                                f_out.write(f"{new_class_id} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}\n")
                
                # Remove empty label files and corresponding images
                if os.path.exists(new_label_path) and os.path.getsize(new_label_path) == 0:
                    os.remove(new_label_path)
                    os.remove(new_img_path)
    
    # Generate dataset.yaml configuration
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n\n")
        f.write(f"nc: {len(ppe_classes)}\n")
        f.write(f"names: {ppe_classes}\n")
    
    return yaml_path


def train_model(yaml_path: str, model_type: str, weights_dir: str, 
                model_name: str, epochs: int, batch_size: int, img_size: int) -> str:
    """
    Train a YOLOv8 model with specified configuration.
    
    Args:
        yaml_path: Path to dataset configuration YAML
        model_type: Base model architecture (e.g., 'yolov8n.pt')
        weights_dir: Directory to save training outputs
        model_name: Name for this training run
        epochs: Number of training epochs
        batch_size: Training batch size
        img_size: Input image size
        
    Returns:
        Path to the best model weights
    """
    model = YOLO(model_type)
    
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=10,
        save=True,
        project=weights_dir,
        name=model_name
    )
    
    return os.path.join(weights_dir, model_name, 'weights', 'best.pt')


def main() -> None:
    """Main execution function for training person and PPE detection models."""
    args = parse_args()
    
    os.makedirs(args.weights_dir, exist_ok=True)
    
    # Prepare person detection dataset
    print("Creating person detection dataset...")
    person_data_dir = os.path.join(args.data_dir, 'person_dataset')
    person_yaml = create_person_dataset(args.data_dir, person_data_dir)
    
    # Prepare PPE detection dataset with cropped person regions
    print("Creating PPE detection dataset with person cropping...")
    ppe_data_dir = os.path.join(args.data_dir, 'ppe_dataset')
    classes_file = os.path.join(args.data_dir, 'classes.txt')
    ppe_yaml = create_ppe_dataset_with_cropping(args.data_dir, ppe_data_dir, classes_file)
    
    # Train person detection model
    print("Training person detection model...")
    person_model_path = train_model(
        person_yaml,
        args.person_model,
        args.weights_dir,
        'person_detection',
        args.epochs,
        args.batch_size,
        args.img_size
    )
    
    # Train PPE detection model
    print("Training PPE detection model...")
    ppe_model_path = train_model(
        ppe_yaml,
        args.ppe_model,
        args.weights_dir,
        'ppe_detection',
        args.epochs,
        args.batch_size,
        args.img_size
    )
    
    print(f"Training complete. Models saved to:")
    print(f"Person detection model: {person_model_path}")
    print(f"PPE detection model: {ppe_model_path}")


if __name__ == "__main__":
    main()
