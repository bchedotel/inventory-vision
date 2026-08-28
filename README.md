# InventoryVision
A computer vision system for detecting, classifying, and analyzing electronic components on PCB boards using YOLOv11.

## Features
- **Component Detection**: Detects 8 types of electronic components (resistors, inductors, capacitors, ICs, diodes, LEDs, connectors, transistors)
- **Classification**: Classifies detected components with confidence scores
- **Mass Estimation**: Estimates component mass based on visual area using statistical models
- **Human Correction**: Interactive tool for validating and correcting detections
- **Data Export**: Generates annotated images and detection reports
- **Model Training**: Fine-tunes a YOLOv11 model on a custom PCB component dataset, with automated dataset splitting, reproducible seeding, and per-split evaluation

## Installation
Requires Python 3.13 and [uv](https://docs.astral.sh/uv/)
```bash
uv sync
```

## Usage

### Inference
```bash
uv run python main.py --input <image_path> --model <model_path> [options]
```
**Required arguments:**
- `--input` - Path(s) to input image(s) (supports multiple files)
- `--model` - Path to the YOLO model (.pt file)

**Optional arguments:**
- `--output-xlsx` - Output Excel file path
- `--debug` - Save debug images with bounding boxes
- `--human` - Enable human-in-the-loop correction (interactive GUI)
- `--metrics` - Display detailed component dimension metrics
- `--bbox-reduction` - Shrink bboxes by percentage (e.g., 4)
- `--bbox-uncertainty` - Uncertainty on bbox dimensions in %
- `--individual-ic` - Output individual IC areas to separate file

**Example:**
```bash
uv run python main.py --input board.jpg --model model/best.pt --debug --human --metrics
```

### Training
```bash
uv run python scripts/train.py [options]
```
Fine-tunes a YOLOv11 model (`yolo11m` by default) on a PCB component dataset: splits the data into train/valid/test, trains with a preconfigured schedule, and evaluates the result on each split.

**Optional arguments:**
- `--dataset` - Path to the source dataset directory, containing `images/`, `labels/`, and `data.yaml` (defaults to `dataset/real_dataset`)
- `--pretrained-model` - Path to pretrained/base model weights to fine-tune from (defaults to `weights/yolo11m.pt`)
- `--cpu` - Train on CPU instead of GPU
- `--output-name` - Custom name for the run's output directory (defaults to `output_yolo11m_seed_<seed>_batch_<batch>`)
- `--seed` - Random seed for reproducibility (default: `14`)

**Expected dataset layout** (source directory passed via `--dataset`):
```
dataset/real_dataset/
├── data.yaml          # or data.yml — must define nc and names
├── images/            # all images, e.g. board_<id>_<n>.png
└── labels/            # YOLO-format .txt labels, one per image
```

## Project Structure
- `main.py` - Main entry point with interactive CLI for detection/inference
- `scripts/` - Core processing modules:
  - `detection_classification.py` - YOLO detection and annotation
  - `crop_board.py` - Board region extraction
  - `component_to_mass.py` - Mass estimation models
  - `transform_detections.py` - Detection format conversion
  - `human_correction.py` - Manual validation interface
  - `train.py` - Training entry point: dataset preparation, YOLOv11 fine-tuning, and evaluation
- `model/best.pt` - Pre-trained YOLOv11 finetuned model
- `weights/yolo11m.pt` - Default base YOLOv11 weights used as the starting point for fine-tuning
- `dataset/real_dataset` - Default source dataset used by `train.py`

## Dependencies
- OpenCV, PyTorch, Ultralytics (YOLO), Pandas, Rich CLI, PyYAML, NumPy
