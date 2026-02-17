#!/usr/bin/env python3
"""
Fine-tune EfficientNetB2 on custom Nest camera bird images.
Uses transfer learning from the existing trained model.

Requires: pip install scikit-learn
"""
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class BirdDataset(Dataset):
    """Custom dataset for bird images organized in folders by species"""

    def __init__(self, root_dir, processor):
        self.root_dir = Path(root_dir)
        self.processor = processor

        # Find all species folders
        self.species_folders = [d for d in self.root_dir.iterdir() if d.is_dir()]
        self.species_to_idx = {folder.name: idx for idx, folder in enumerate(sorted(self.species_folders))}
        self.idx_to_species = {idx: name for name, idx in self.species_to_idx.items()}

        # Collect all images
        self.images = []
        for species_folder in self.species_folders:
            species_name = species_folder.name
            species_idx = self.species_to_idx[species_name]

            for img_path in species_folder.glob("*.jpg"):
                self.images.append((img_path, species_idx))

        print(f"  Loaded {len(self.images)} images from {len(self.species_folders)} species")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Process with HuggingFace processor
        inputs = self.processor(images=image, return_tensors="pt")

        # Remove batch dimension
        pixel_values = inputs['pixel_values'].squeeze(0)

        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    """Compute accuracy and F1 metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_custom_model(
    epochs=10,
    batch_size=8,
    learning_rate=2e-5,
    base_model="dennisjooo/Birds-Classifier-EfficientNetB2"
):
    """
    Fine-tune bird classifier on custom dataset.

    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for fine-tuning
        base_model: Pretrained model to start from
    """

    training_dir = Path("training_data")
    train_dir = training_dir / "train"
    val_dir = training_dir / "val"

    # Check dataset exists
    if not train_dir.exists() or not val_dir.exists():
        print(f"❌ Training data not found at {training_dir}")
        print(f"   Run prepare_training_data.py first")
        return

    # Load dataset metadata
    metadata_file = training_dir / "dataset_info.json"
    with open(metadata_file, 'r') as f:
        dataset_info = json.load(f)

    num_species = dataset_info['num_species']
    species_names = dataset_info['species']

    print(f"\n{'='*70}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Base model: {base_model}")
    print(f"Species: {num_species}")
    print(f"Training images: {dataset_info['train_images']}")
    print(f"Validation images: {dataset_info['val_images']}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")

    # Load processor and model
    print(f"\n{'='*70}")
    print("LOADING BASE MODEL")
    print(f"{'='*70}")

    processor = AutoImageProcessor.from_pretrained(base_model)
    model = AutoModelForImageClassification.from_pretrained(
        base_model,
        num_labels=num_species,
        ignore_mismatched_sizes=True  # Since we're changing number of classes
    )

    # Update label mappings
    model.config.id2label = {idx: name for idx, name in enumerate(species_names)}
    model.config.label2id = {name: idx for idx, name in enumerate(species_names)}

    print(f"✓ Loaded model with {num_species} output classes")

    # Create datasets
    print(f"\n{'='*70}")
    print("LOADING TRAINING DATA")
    print(f"{'='*70}")

    train_dataset = BirdDataset(train_dir, processor)
    val_dataset = BirdDataset(val_dir, processor)

    # Training arguments
    output_dir = Path("custom_model")
    output_dir.mkdir(exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        remove_unused_columns=False,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}")

    trainer.train()

    # Save final model
    final_model_path = output_dir / "final_model"
    trainer.save_model(str(final_model_path))
    processor.save_pretrained(str(final_model_path))

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Model saved to {final_model_path}")

    # Evaluate
    print(f"\nFinal evaluation:")
    metrics = trainer.evaluate()
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Save training info
    training_info = {
        'base_model': base_model,
        'num_species': num_species,
        'species': species_names,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'final_metrics': metrics,
        'model_path': str(final_model_path)
    }

    with open(output_dir / "training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)

    print(f"\n✓ Training info saved to {output_dir / 'training_info.json'}")
    print(f"\nTo use this model, update hf_species_classifier.py:")
    print(f"  MODEL_NAME = \"{final_model_path}\"")

if __name__ == "__main__":
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cpu":
        print("⚠️  Training on CPU will be slow. Consider using GPU.")
        print("   You can reduce batch_size to save memory.")

    train_custom_model(
        epochs=10,
        batch_size=4,  # Small batch size for CPU training
        learning_rate=2e-5
    )
