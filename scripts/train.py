import shutil
import random
import yaml
from ultralytics import YOLO
from pathlib import Path
import re
import json
from datetime import datetime
import argparse
import os
import numpy as np

# must be set before torch is imported
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

import torch
torch.cuda.empty_cache()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seeds set to {seed} (deterministic)")


def get_custom_augmentations():
    """Augmentation params for model.train() (see Ultralytics docs for full arg reference)."""
    return {
        'mosaic': 1.0,
        'copy_paste': 0.5,
        'mixup': 0.2,

        'degrees': 10.0,
        'translate': 0.15,
        'scale': 0.7,
        'shear': 2.0,
        'perspective': 0.0,      # deforms small objects, keep off

        'fliplr': 0.5,
        'flipud': 0.0,           # PCBs have a fixed orientation

        'hsv_h': 0.015,
        'hsv_s': 0.5,
        'hsv_v': 0.3,

        'erasing': 0.0,          # can erase small components
    }


NAME = "finetuned"
WORKERS = 4
IMG_SIZE = 1600
BATCH = -1  # Auto-detect
EPOCHS = 400
LR0 = 0.002
LRF = 0.00005
PATIENCE = 100
OPTIMIZER = "AdamW"
WEIGHT_DECAY = 0.003
DROPOUT = 0.2
CLOSE_MOSAIC = 50
WARMUP_EPOCHS = 10
WARMUP_MOMENTUM = 0.8
MOMENTUM = 0.937

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "weights" / "yolo11m.pt"
DEFAULT_DATASET_DIR = PROJECT_DIR / "dataset" / "real_dataset"

SPLIT_RATIOS = {"train": 0.7, "valid": 0.2, "test": 0.1}


def save_hyperparameters(save_path: Path, seed: int, stage: str, **params):
    """Save hyperparameters to a JSON file"""
    config = {
        "stage": stage,
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": params,
        "reproducibility": {
            "seed": seed,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Hyperparameters saved to {save_path}")


def count_images(split_dir: Path) -> int:
    """Count images in a split directory, returning 0 if it doesn't exist"""
    return len(list(split_dir.glob("*.*"))) if split_dir.exists() else 0


def analyze_training_schedule(dataset_dir: Path, batch_size: int, epochs: int):
    """Analyze training iterations and provide recommendations"""
    train_images = count_images(dataset_dir / "train" / "images")
    val_images = count_images(dataset_dir / "valid" / "images")
    test_images = count_images(dataset_dir / "test" / "images")

    if batch_size == -1:
        batch_size = 20  # Expected auto-detect value for yolo11m

    iters_per_epoch = np.ceil(train_images / batch_size)
    total_iters = iters_per_epoch * epochs

    print("\n" + "=" * 70)
    print("TRAINING SCHEDULE ANALYSIS")
    print("=" * 70)
    print(f"Dataset: {train_images} train | {val_images} val | {test_images} test")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print("-" * 70)
    print(f"Iterations/epoch: {int(iters_per_epoch)}")
    print(f"Total iterations: {int(total_iters)}")
    print("-" * 70)

def parse_filtered_lines_to_bboxes_and_ids(lines):
    bboxes, cids = [], []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            cids.append(int(parts[0]))
            bboxes.append([float(x) for x in parts[1:]])
    return bboxes, cids


def extract_board_id(filename: str):
    match = re.match(r"^board_(\d+)_(\d+)", filename)
    return match.group(1) if match else None


def write_yolo_label_file(path: Path, bboxes, cids):
    lines = [f"{cid} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}" for cid, bb in zip(cids, bboxes)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fw:
        fw.write("\n".join(lines) + ("\n" if lines else ""))


def prepare_dataset(source_dir: Path, output_dir: Path, split_ratios: dict = None):
    """Prepare dataset with train/val/test splits, grouped by board ID"""
    if split_ratios is None:
        split_ratios = SPLIT_RATIOS

    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in split_ratios:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    samples = []
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"

    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.JPG', '*.JPEG']:
        image_files.extend(images_dir.glob(ext))

    image_files = sorted(image_files)

    for img_path in image_files:
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            print(f"Warning: missing label file for {img_path.name}")
            continue
        with open(lbl_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        samples.append((img_path, lbl_path, lines))

    groups = {}
    no_board_group_key = None

    for sample in samples:
        img_path = sample[0]
        board_id = extract_board_id(img_path.stem)
        key = board_id if board_id else no_board_group_key
        groups.setdefault(key, []).append(sample)

    print(f"Grouped samples into {len(groups)} groups (board ID-based).")

    group_items = sorted(list(groups.items()), key=lambda x: str(x[0]))
    random.shuffle(group_items)
    allocated = {k: [] for k in split_ratios}
    pos = 0

    for split, ratio in split_ratios.items():
        if split == list(split_ratios.keys())[-1]:
            split_groups = group_items[pos:]
        else:
            target = int(round(len(group_items) * ratio))
            split_groups = group_items[pos:pos + target]
            pos += target
        for key, items in split_groups:
            allocated[split].extend(items)

    for split, subset in allocated.items():
        print(f"Preparing split '{split}' with {len(subset)} samples")
        subset_sorted = sorted(subset, key=lambda x: str(x[0]))
        for img_src, lbl_src, filtered in subset_sorted:
            dst_img = output_dir / split / "images" / img_src.name
            dst_lbl = output_dir / split / "labels" / lbl_src.name
            shutil.copy(img_src, dst_img)
            write_yolo_label_file(dst_lbl, *parse_filtered_lines_to_bboxes_and_ids(filtered))

    return output_dir


def create_yaml_config(dataset_dir: Path, yaml_config: dict, output_yaml: Path):
    """Create YAML config for dataset"""
    config_dict = {
        "path": str(dataset_dir.parent),
        "train": str(dataset_dir / "train" / "images"),
        "val": str(dataset_dir / "valid" / "images"),
        "nc": yaml_config["nc"],
        "names": yaml_config.get("names", [])
    }

    if (dataset_dir / "test" / "images").exists():
        config_dict["test"] = str(dataset_dir / "test" / "images")

    with open(output_yaml, "w") as f:
        yaml.safe_dump(config_dict, f)

    return output_yaml


def organize_evaluation_results(model, normalized_yaml, eval_dir: Path):
    print(f"\nOrganizing evaluation results in {eval_dir}...")

    with open(normalized_yaml, "r") as f:
        yaml_content = yaml.safe_load(f)

    for split in ["train", "val", "test"]:
        (eval_dir / split).mkdir(parents=True, exist_ok=True)

        print(f"\nEvaluating on {split} set...")
        split_key = split if split != "val" else "valid"
        split_path = yaml_content.get(split, yaml_content.get(split_key))

        if split_path and Path(split_path).exists():
            try:
                metrics = model.val(
                    data=str(normalized_yaml),
                    split=split,
                    save_json=True,
                    save_hybrid=True,
                    project=str(eval_dir / split),
                    name="results",
                    exist_ok=True
                )

                metrics_file = eval_dir / split / "metrics.json"
                with open(metrics_file, "w") as f:
                    json.dump({
                        "split": split,
                        "map50": float(metrics.box.map50) if hasattr(metrics.box, 'map50') else None,
                        "map50_95": float(metrics.box.map) if hasattr(metrics.box, 'map') else None,
                        "precision": float(metrics.box.mp) if hasattr(metrics.box, 'mp') else None,
                        "recall": float(metrics.box.mr) if hasattr(metrics.box, 'mr') else None,
                    }, f, indent=2)
                print(f"{split} evaluation completed")
            except Exception as e:
                print(f"Error evaluating {split} set: {e}")
        else:
            print(f"{split} path not found")


def main(args):
    DEVICE = "cpu" if args.cpu else "cuda"

    SEED = args.seed
    set_seed(SEED)

    model_path = Path(args.pretrained_model) if args.pretrained_model else DEFAULT_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Pretrained model not found: {model_path}")
    print(f"Using pretrained model: {model_path}")

    if args.output_name:
        OUTPUT_DIR = PROJECT_DIR / args.output_name
    else:
        OUTPUT_DIR = PROJECT_DIR / f"output_yolo11m_seed_{SEED}_batch_{BATCH}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(args.dataset) if args.dataset else DEFAULT_DATASET_DIR
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    data_yaml = next(
        (p for p in [dataset_dir / "data.yaml", dataset_dir / "data.yml"] if p.exists()), None
    )
    if not data_yaml:
        raise FileNotFoundError(f"No data.yaml found in {dataset_dir}")

    with open(data_yaml) as f:
        yaml_config = yaml.safe_load(f)

    print("\n" + "=" * 80)
    print("PREPARING DATASET WITH TRAIN/VAL/TEST SPLIT")
    print("=" * 80)

    dataset_prepared = OUTPUT_DIR / "dataset_prepared"
    prepare_dataset(dataset_dir, dataset_prepared, split_ratios=SPLIT_RATIOS)

    data_yaml_prepared = OUTPUT_DIR / "data.yaml"
    create_yaml_config(dataset_prepared, yaml_config, data_yaml_prepared)

    print("\n" + "=" * 80)
    print("FINE-TUNING")
    print("=" * 80)

    analyze_training_schedule(dataset_prepared, BATCH if BATCH > 0 else 20, EPOCHS)

    # Shared hyperparameters, reused for both the saved config and model.train()
    train_kwargs = dict(
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        weight_decay=WEIGHT_DECAY,
        dropout=DROPOUT,
        momentum=MOMENTUM,
        patience=PATIENCE,
        close_mosaic=CLOSE_MOSAIC,
        workers=WORKERS,
    )

    finetune_eval_dir = OUTPUT_DIR / "evaluation" / "finetuned"
    save_hyperparameters(
        finetune_eval_dir / "hyperparameters.json",
        SEED,
        stage="finetuning",
        base_model=str(model_path),
        split_ratios=SPLIT_RATIOS,
        **train_kwargs
    )

    model = YOLO(str(model_path))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    cache_files = list(dataset_prepared.rglob("*.cache"))
    for cache in cache_files:
        try:
            cache.unlink()
        except Exception as e:
            print(f"Warning: Could not remove {cache}: {e}")

    aug_params = get_custom_augmentations()

    try:
        model.train(
            data=str(data_yaml_prepared),
            project=OUTPUT_DIR,
            name=NAME,
            device=DEVICE,
            exist_ok=True,
            save_dir=OUTPUT_DIR / NAME,
            warmup_epochs=WARMUP_EPOCHS,
            warmup_momentum=WARMUP_MOMENTUM,
            seed=SEED,
            deterministic=True,
            cache='disk',
            amp=True,
            box=7.5,
            cls=1,
            dfl=1.5,
            **train_kwargs,
            **aug_params
        )
    except KeyboardInterrupt:
        print("\nFine-tuning interrupted. Proceeding to evaluation.")

    print("\n" + "=" * 80)
    print("EVALUATING FINE-TUNED MODEL (TRAIN/VAL/TEST)")
    print("=" * 80)
    organize_evaluation_results(model, data_yaml_prepared, finetune_eval_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a YOLO model on a dataset in YOLO format."
    )
    parser.add_argument("--dataset", type=str, default=None,
                         help=f"Path to the dataset directory containing images/, labels/, and data.yaml "
                              f"(defaults to {DEFAULT_DATASET_DIR}).")
    parser.add_argument("--pretrained-model", type=str, default=None,
                         help="Path to pretrained model weights (defaults to weights/yolo11m.pt).")
    parser.add_argument("--cpu", action="store_true",
                         help="Train on CPU instead of GPU.")
    parser.add_argument("--output-name", type=str, default=None,
                         help="Custom name for the output directory.")
    parser.add_argument("--seed", type=int, default=14,
                         help="Random seed for reproducibility (default: 14)")
    args = parser.parse_args()
    main(args)
