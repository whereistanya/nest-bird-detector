#!/usr/bin/env python3
"""
Prepare labeled images for training by:
1. Reading manual labels
2. Cropping to bounding boxes (using detection logs)
3. Organizing into train/val splits by species
4. Creating dataset metadata
"""
import json
import csv
from pathlib import Path
from PIL import Image
from collections import defaultdict
import random

def prepare_training_data(train_split=0.8, min_images_per_species=5):
    """
    Prepare training dataset from labeled images.

    Args:
        train_split: Fraction of data to use for training (rest is validation)
        min_images_per_species: Minimum images required per species
    """

    eval_dir = Path("evaluation_dataset")
    labels_file = eval_dir / "manual_labels.json"
    frames_dir = eval_dir / "frames"

    training_dir = Path("training_data")
    training_dir.mkdir(exist_ok=True)

    train_dir = training_dir / "train"
    val_dir = training_dir / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    # Load labels
    labels = {}

    if labels_file.exists():
        with open(labels_file, 'r') as f:
            labels = json.load(f)
        print(f"✓ Loaded {len(labels)} manual labels")
    else:
        print(f"⚠️  No manual labels found at {labels_file}")

    # Load corrections from feedback loop
    corrections_file = Path("bird_corrections.json")
    if corrections_file.exists():
        with open(corrections_file, 'r') as f:
            corrections = json.load(f)

        # Merge corrections into labels (corrections override manual labels)
        corrections_added = 0
        corrections_updated = 0

        for img_path, correction in corrections.items():
            # Extract filename from path
            filename = Path(img_path).name

            # Skip "Not a Bird" and "Unknown" corrections (not useful for training)
            corrected_species = correction['corrected_species']
            if corrected_species in ['Not a Bird', 'Unknown']:
                continue

            # Create label entry
            label_entry = {
                'species': corrected_species,
                'quality': 'good',
                'source': 'user_correction',
                'timestamp': correction.get('timestamp', '')
            }

            if filename in labels:
                corrections_updated += 1
            else:
                corrections_added += 1

            labels[filename] = label_entry

        print(f"✓ Loaded {len(corrections)} corrections from feedback loop")
        print(f"  Added {corrections_added} new labels")
        print(f"  Updated {corrections_updated} existing labels")
    else:
        print(f"⚠️  No corrections found at {corrections_file}")

    if not labels:
        print(f"\n❌ No labels available (neither manual nor corrections)")
        print(f"   Run label_images.py or use the feedback UI to label images")
        return

    print(f"\n✓ Total labels available: {len(labels)}")

    # Load bounding boxes from detection logs
    detections_csv = Path("logs/detections.csv")
    bboxes = {}

    if detections_csv.exists():
        with open(detections_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = Path(row['snapshot_file']).name
                bboxes[filename] = (
                    int(row['bbox_x1']),
                    int(row['bbox_y1']),
                    int(row['bbox_x2']),
                    int(row['bbox_y2'])
                )
        print(f"✓ Loaded {len(bboxes)} bounding boxes")

    # Organize images by species
    species_images = defaultdict(list)

    for filename, label_data in labels.items():
        if label_data['quality'] != 'good':
            continue  # Skip unknown/bad images

        species = label_data['species']

        # Try multiple locations for the image
        # 1. Evaluation frames directory
        # 2. Snapshots directory (for corrections)
        # 3. Direct path (if filename is actually a full path)
        possible_paths = [
            frames_dir / filename,
            Path("snapshots") / filename,
            Path(filename)
        ]

        img_path = None
        for path in possible_paths:
            if path.exists():
                img_path = path
                break

        if img_path is None:
            print(f"⚠️  Image not found: {filename}")
            continue

        bbox = bboxes.get(filename)
        species_images[species].append({
            'filename': filename,
            'path': img_path,
            'bbox': bbox
        })

    # Filter species with enough images
    print(f"\n{'='*70}")
    print("SPECIES DISTRIBUTION")
    print(f"{'='*70}")

    valid_species = {}
    for species, images in sorted(species_images.items()):
        count = len(images)
        status = "✓" if count >= min_images_per_species else "✗"
        print(f"{status} {species}: {count} images")

        if count >= min_images_per_species:
            valid_species[species] = images

    if not valid_species:
        print(f"\n❌ No species have >= {min_images_per_species} images")
        print(f"   Collect more images or reduce min_images_per_species")
        return

    print(f"\n✓ {len(valid_species)} species with sufficient data")

    # Split into train/val and crop images
    print(f"\n{'='*70}")
    print("PREPARING TRAINING DATA")
    print(f"{'='*70}")

    dataset_info = {
        'species': list(valid_species.keys()),
        'num_species': len(valid_species),
        'train_images': 0,
        'val_images': 0,
        'train_split': train_split
    }

    for species, images in valid_species.items():
        # Create species folders
        species_name_safe = species.replace(' ', '_').replace('-', '_').lower()
        train_species_dir = train_dir / species_name_safe
        val_species_dir = val_dir / species_name_safe
        train_species_dir.mkdir(exist_ok=True)
        val_species_dir.mkdir(exist_ok=True)

        # Shuffle and split
        random.shuffle(images)
        split_idx = int(len(images) * train_split)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        print(f"\n{species}:")
        print(f"  Train: {len(train_images)} images")
        print(f"  Val: {len(val_images)} images")

        # Process training images
        for img_data in train_images:
            img = Image.open(img_data['path'])

            # Crop to bbox if available
            if img_data['bbox']:
                img = img.crop(img_data['bbox'])

            # Save with unique name
            output_path = train_species_dir / img_data['filename']
            img.save(output_path)
            dataset_info['train_images'] += 1

        # Process validation images
        for img_data in val_images:
            img = Image.open(img_data['path'])

            # Crop to bbox if available
            if img_data['bbox']:
                img = img.crop(img_data['bbox'])

            # Save with unique name
            output_path = val_species_dir / img_data['filename']
            img.save(output_path)
            dataset_info['val_images'] += 1

    # Save dataset metadata
    metadata_file = training_dir / "dataset_info.json"
    with open(metadata_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\n{'='*70}")
    print("DATASET READY FOR TRAINING")
    print(f"{'='*70}")
    print(f"Species: {dataset_info['num_species']}")
    print(f"Training images: {dataset_info['train_images']}")
    print(f"Validation images: {dataset_info['val_images']}")
    print(f"\nDataset structure:")
    print(f"  {training_dir}/")
    print(f"    train/")
    for species in valid_species.keys():
        species_name_safe = species.replace(' ', '_').replace('-', '_').lower()
        print(f"      {species_name_safe}/")
    print(f"    val/")
    for species in valid_species.keys():
        species_name_safe = species.replace(' ', '_').replace('-', '_').lower()
        print(f"      {species_name_safe}/")
    print(f"\n✓ Metadata saved to {metadata_file}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    prepare_training_data()
