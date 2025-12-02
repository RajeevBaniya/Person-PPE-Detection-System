#!/usr/bin/env python3
import os
import argparse
import xml.etree.ElementTree as ET
import glob
from pathlib import Path
from typing import List, Dict, Optional
import shutil


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for dataset conversion."""
    parser = argparse.ArgumentParser(description='Convert PascalVOC annotations to YOLOv8 format')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the directory containing PascalVOC annotations')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save the YOLOv8 annotations')
    parser.add_argument('--classes_file', type=str, default=None, help='Path to a file containing class names')
    parser.add_argument('--images_dir', type=str, required=True, help='Path to the directory containing images')
    return parser.parse_args()


def get_classes(annotation_dir: str, classes_file: Optional[str] = None) -> List[str]:
    """
    Extract class names from annotations or load from file.
    
    Args:
        annotation_dir: Directory containing PascalVOC XML annotations
        classes_file: Optional path to file with predefined class names
        
    Returns:
        Sorted list of class names
    """
    if classes_file and os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        return classes
    
    # Extract classes from XML annotations
    classes = set()
    for xml_file in glob.glob(os.path.join(annotation_dir, '*.xml')):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall('.//object'):
            class_name_elem = obj.find('n')
            if class_name_elem is not None and class_name_elem.text:
                classes.add(class_name_elem.text)
            else:
                class_name_elem = obj.find('name')
                if class_name_elem is not None and class_name_elem.text:
                    classes.add(class_name_elem.text)
    return sorted(list(classes))


def create_yolo_folders(output_dir: str) -> None:
    """
    Create directory structure for YOLOv8 dataset.
    
    Args:
        output_dir: Root directory for the dataset
    """
    os.makedirs(os.path.join(output_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'labels'), exist_ok=True)

def convert_annotation(xml_file: str, output_path: str, class_map: Dict[str, int]) -> None:
    """
    Convert a single PascalVOC XML annotation to YOLOv8 format.
    
    Args:
        xml_file: Path to input XML annotation file
        output_path: Directory to save converted annotation
        class_map: Mapping from class names to class IDs
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Extract image dimensions
    size = root.find('.//size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    
    # Convert each object annotation
    stem = Path(xml_file).stem
    with open(os.path.join(output_path, f"{stem}.txt"), 'w') as out_file:
        for obj in root.findall('.//object'):
            # Try to get class name from 'n' tag first, then from 'name' tag
            class_name_elem = obj.find('n')
            if class_name_elem is not None and class_name_elem.text:
                class_name = class_name_elem.text
            else:
                class_name_elem = obj.find('name')
                if class_name_elem is not None and class_name_elem.text:
                    class_name = class_name_elem.text
                else:
                    continue
            
            if class_name not in class_map:
                continue
                
            # Extract bounding box coordinates
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Convert to YOLO format (normalized center coordinates and dimensions)
            x_center = (xmin + xmax) / 2.0 / width
            y_center = (ymin + ymax) / 2.0 / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height
            
            # Write annotation line
            class_id = class_map[class_name]
            out_file.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")

def split_dataset(annotation_files: List[str], images_dir: str, output_dir: str, 
                  class_map: Dict[str, int], train_ratio: float = 0.8, 
                  val_ratio: float = 0.1, test_ratio: float = 0.1) -> None:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        annotation_files: List of paths to XML annotation files
        images_dir: Directory containing source images
        output_dir: Root output directory
        class_map: Mapping from class names to class IDs
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
    """
    import random
    random.shuffle(annotation_files)
    
    # Calculate split points
    train_end = int(len(annotation_files) * train_ratio)
    val_end = train_end + int(len(annotation_files) * val_ratio)
    
    # Split file lists
    train_files = annotation_files[:train_end]
    val_files = annotation_files[train_end:val_end]
    test_files = annotation_files[val_end:]
    
    print(f"Splitting dataset: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test")
    
    # Process each split
    for xml_file in train_files:
        process_file(xml_file, images_dir, os.path.join(output_dir, 'train'), class_map)
    
    for xml_file in val_files:
        process_file(xml_file, images_dir, os.path.join(output_dir, 'val'), class_map)
    
    for xml_file in test_files:
        process_file(xml_file, images_dir, os.path.join(output_dir, 'test'), class_map)


def process_file(xml_file: str, images_dir: str, output_dir: str, class_map: Dict[str, int]) -> None:
    """
    Process a single annotation file and copy its corresponding image.
    
    Args:
        xml_file: Path to XML annotation file
        images_dir: Directory containing source images
        output_dir: Output directory for this split
        class_map: Mapping from class names to class IDs
    """
    convert_annotation(xml_file, os.path.join(output_dir, 'labels'), class_map)
    
    # Copy corresponding image
    image_filename = Path(xml_file).stem + '.jpg'
    src_image = os.path.join(images_dir, image_filename)
    if os.path.exists(src_image):
        shutil.copy(src_image, os.path.join(output_dir, 'images', image_filename))
    else:
        print(f"Warning: Image {image_filename} not found in {images_dir}")

def main() -> None:
    """Main execution function for converting PascalVOC to YOLO format."""
    args = parse_args()
    
    annotation_dir = args.input_dir
    output_dir = args.output_dir
    images_dir = args.images_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directory structure
    create_yolo_folders(output_dir)
    
    # Collect all XML annotation files
    annotation_files = glob.glob(os.path.join(annotation_dir, '*.xml'))
    if not annotation_files:
        print(f"No XML files found in {annotation_dir}")
        return
    
    # Extract or load class names
    classes = get_classes(annotation_dir, args.classes_file)
    class_map = {class_name: i for i, class_name in enumerate(classes)}
    
    print(f"Found {len(classes)} classes: {classes}")
    
    # Save class names
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        for class_name in classes:
            f.write(f"{class_name}\n")
    
    # Generate YAML configuration for YOLOv8
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n\n")
        f.write(f"nc: {len(classes)}\n")
        f.write(f"names: {[c for c in classes]}\n")
    
    # Split and convert dataset
    split_dataset(annotation_files, images_dir, output_dir, class_map)
    
    print(f"Conversion complete. Output saved to {output_dir}")


if __name__ == "__main__":
    main()
