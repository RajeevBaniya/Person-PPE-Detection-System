# Person and PPE (Personal Protective Equipment) Detection with YOLOv8

This project implements a two-stage detection system using YOLOv8 for person detection and PPE(Personal Protective Equipment) item detection.

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── scripts/
│   ├── pascalVOC_to_yolo.py  # Convert PascalVOC to YOLO format
│   ├── train_models.py       # Train person and PPE detection models
│   └── inference.py          # Run inference with trained models
├── weights/                  # Trained model weights
│   ├── person_detection/     
│   └── ppe_detection/        
└── requirements.txt          # Project dependencies
```

## Usage Instructions

### 1. Convert PascalVOC to YOLO Format

```bash
python scripts/pascalVOC_to_yolo.py --input_dir <input_dir> --output_dir <output_dir> --images_dir <images_dir> --classes_file <classes_file>
```

### 2. Train Models

```bash
python scripts/train_models.py --data_dir <yolo_dataset> --weights_dir weights --epochs 50 --batch_size 16 --img_size 640
```

### 3. Run Inference

```bash
python scripts/inference.py --input_dir <input_dir> --output_dir <output_dir> --person_det_model weights/person_detection/weights/best.pt --ppe_detection_model weights/ppe_detection/weights/best.pt
```


## Implementation Details

1. **Two-Stage Detection**:
   - First stage: Person detection on full images
   - Second stage: PPE detection on cropped person regions

2. **PPE Classes**:
   - hard-hat
   - gloves
   - boots
   - vest
   - ppe-suit

## Model Performance

PPE Detection Performance:
- Overall: mAP50 = 0.334, mAP50-95 = 0.199
- hard-hat: mAP50 = 0.821 (Best performing)
- gloves: mAP50 = 0.088
- boots: mAP50 = 0.342
- vest: mAP50 = 0.073
- ppe-suit: mAP50 = 0.345
