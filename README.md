# Inventory Vision

A computer vision system for detecting, classifying, and analyzing electronic components on PCB boards using YOLOv11.

## Features

- **Component Detection**: Detects 8 types of electronic components (resistors, inductors, capacitors, ICs, diodes, LEDs, connectors, transistors)
- **Classification**: Classifies detected components with confidence scores
- **Mass Estimation**: Estimates component mass based on visual area using statistical models
- **Human Correction**: Interactive tool for validating and correcting detections
- **Data Export**: Generates annotated images and detection reports

## Installation

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/)

```bash
uv sync
```

## Usage

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

## Project Structure

- `main.py` - Main entry point with interactive CLI
- `scripts/` - Core processing modules:
  - `detection_classification.py` - YOLO detection and annotation
  - `crop_board.py` - Board region extraction
  - `component_to_mass.py` - Mass estimation models
  - `transform_detections.py` - Detection format conversion
  - `human_correction.py` - Manual validation interface
- `model/best.pt` - Pre-trained YOLOv11 finetuned model

## Dependencies

- OpenCV, PyTorch, Ultralytics (YOLO), Pandas, Rich CLI