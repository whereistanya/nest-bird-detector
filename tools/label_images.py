#!/usr/bin/env python3
"""
Interactive image labeling tool for building training dataset.
Shows each image and prompts for the correct species label.
"""
import json
from pathlib import Path
from PIL import Image
import sys

def show_image_info(image_path):
    """Display image info"""
    img = Image.open(image_path)
    print(f"\nImage: {image_path.name}")
    print(f"Size: {img.width}x{img.height}")
    print(f"Opening in default viewer...")

    # Open image in default viewer
    try:
        img.show()
    except Exception:
        print("Could not open image viewer")

def label_images():
    """Interactive labeling session"""

    eval_dir = Path("evaluation_dataset")
    frames_dir = eval_dir / "frames"
    labels_file = eval_dir / "manual_labels.json"

    # Load existing labels if any
    if labels_file.exists():
        with open(labels_file, 'r') as f:
            labels = json.load(f)
        print(f"✓ Loaded {len(labels)} existing labels")
    else:
        labels = {}

    # Get all images
    images = sorted(frames_dir.glob("bird_*.jpg"))
    unlabeled = [img for img in images if img.name not in labels]

    print(f"\n{'='*70}")
    print(f"IMAGE LABELING TOOL")
    print(f"{'='*70}")
    print(f"Total images: {len(images)}")
    print(f"Already labeled: {len(labels)}")
    print(f"Remaining: {len(unlabeled)}")
    print(f"\nCommon New Mexico birds:")
    print("  - House Finch")
    print("  - Mourning Dove")
    print("  - Northern Flicker")
    print("  - American Robin")
    print("  - Black-chinned Hummingbird")
    print("  - Lesser Goldfinch")
    print("  - White-winged Dove")
    print("\nCommands:")
    print("  - Enter species name (e.g., 'House Finch')")
    print("  - 'skip' to skip this image")
    print("  - 'unknown' if you can't identify it")
    print("  - 'bad' if image is too blurry/unclear")
    print("  - 'quit' to save and exit")

    for img_path in unlabeled:
        print(f"\n{'='*70}")
        show_image_info(img_path)

        while True:
            label = input(f"\nSpecies (or command): ").strip()

            if label.lower() == 'quit':
                print("\n✓ Saving labels and exiting...")
                with open(labels_file, 'w') as f:
                    json.dump(labels, f, indent=2)
                print(f"✓ Saved {len(labels)} labels to {labels_file}")
                return

            if label.lower() == 'skip':
                print("Skipped")
                break

            if label:
                # Save label
                labels[img_path.name] = {
                    'species': label.title(),
                    'quality': 'good' if label.lower() not in ['unknown', 'bad'] else label.lower()
                }

                # Auto-save after each label
                with open(labels_file, 'w') as f:
                    json.dump(labels, f, indent=2)

                print(f"✓ Labeled as: {label.title()}")
                break

    print(f"\n{'='*70}")
    print(f"LABELING COMPLETE!")
    print(f"{'='*70}")
    print(f"Total labeled: {len(labels)}")

    # Show species distribution
    species_counts = {}
    for data in labels.values():
        if data['quality'] == 'good':
            species = data['species']
            species_counts[species] = species_counts.get(species, 0) + 1

    print(f"\nSpecies distribution:")
    for species, count in sorted(species_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {species}: {count}")

    with open(labels_file, 'w') as f:
        json.dump(labels, f, indent=2)

    print(f"\n✓ Labels saved to {labels_file}")

if __name__ == "__main__":
    try:
        label_images()
    except KeyboardInterrupt:
        print("\n\n✓ Interrupted. Labels have been auto-saved.")
        sys.exit(0)
