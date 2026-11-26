# =============================================================================
# Vision Mamba (Vim) Fine-Tuning for FER2013 - Complete Kaggle GPU Ready
# =============================================================================
# Copy-paste ready! Run this directly on Kaggle GPU
# No repo cloning needed. Uses HuggingFace directly.

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================

# Uncomment and run this cell first
"""
!pip install --upgrade transformers datasets accelerate torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install scikit-learn matplotlib pillow -q
"""

# ============================================================================
# CELL 2: Imports
# ============================================================================

import os
import torch
import numpy as np
import json
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer
)
from torchvision.transforms import (
    CenterCrop, Compose, Normalize, RandomHorizontalFlip,
    RandomResizedCrop, ToTensor, Resize, RandomRotation, ColorJitter
)
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from PIL import Image

print("✅ All imports successful!")

# ============================================================================
# CELL 3: GPU Setup & Configuration
# ============================================================================

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Device: {device}")

if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"✓ CUDA Version: {torch.version.cuda}")
else:
    print("⚠️  GPU not available! Training will be slow on CPU")

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# CELL 4: Load FER2013 Dataset
# ============================================================================

print("\n" + "="*60)
print("LOADING FER2013 DATASET")
print("="*60)

data_dir = '/kaggle/input/fer2013'

# Check if path exists
if not os.path.exists(data_dir):
    print(f"❌ Error: {data_dir} not found!")
    print("Make sure FER2013 dataset is added to Kaggle notebook")
else:
    print(f"✓ Dataset path: {data_dir}")

# Load dataset
print("Loading dataset...")
dataset = load_dataset('imagefolder', data_dir=data_dir)
print(f"✓ Dataset loaded: {dataset}")

# ============================================================================
# CELL 5: Create Train/Val/Test Splits
# ============================================================================

print("\n" + "="*60)
print("CREATING DATA SPLITS")
print("="*60)

train_ds_full = dataset['train']
test_ds = dataset['test']

# Split training into train and validation (90/10)
train_val_split = train_ds_full.train_test_split(test_size=0.1, seed=42)
train_ds = train_val_split['train']
val_ds = train_val_split['test']

print(f"✓ Training samples:   {len(train_ds)}")
print(f"✓ Validation samples: {len(val_ds)}")
print(f"✓ Test samples:       {len(test_ds)}")

# ============================================================================
# CELL 6: Extract Labels
# ============================================================================

print("\n" + "="*60)
print("EMOTION CLASSES")
print("="*60)

labels = train_ds.features['label'].names
print(f"✓ Number of classes: {len(labels)}")
print(f"✓ Classes: {labels}")

# Create label mappings
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in id2label.items()}
num_labels = len(labels)

# Print mapping
for label_id, label_name in id2label.items():
    print(f"  {label_id}: {label_name}")

# ============================================================================
# CELL 7: Visualize Sample Images
# ============================================================================

print("\n" + "="*60)
print("VISUALIZING SAMPLE IMAGES")
print("="*60)

def visualize_samples(ds, labels, filename='vim_samples.png'):
    """Visualize one sample per emotion class"""
    shown = set()
    fig, axs = plt.subplots(1, len(labels), figsize=(16, 3))
    
    for sample in ds:
        label_name = labels[sample["label"]]
        if label_name not in shown:
            idx = len(shown)
            if idx < len(labels):
                axs[idx].imshow(sample["image"])
                axs[idx].set_title(label_name, fontsize=10, fontweight='bold')
                axs[idx].axis("off")
                shown.add(label_name)
        if len(shown) == len(labels):
            break
    
    plt.suptitle("Sample Images per Emotion Class", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.show()

visualize_samples(train_ds, labels, filename='/kaggle/working/vim_samples.png')

# ============================================================================
# CELL 8: Setup Model & Processor
# ============================================================================

print("\n" + "="*60)
print("MODEL & PROCESSOR SETUP")
print("="*60)

model_name = "hustvl/vim-tiny-midclstok"
print(f"✓ Model: {model_name}")
print(f"✓ Model size: ~7M parameters (pretrained on ImageNet)")

# Load processor
processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
print(f"✓ Processor loaded")

# Get image size and normalization values
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]
size = 224

print(f"✓ Image size: {size}x{size}")
print(f"✓ Normalization: ImageNet (mean={image_mean}, std={image_std})")

# ============================================================================
# CELL 9: Define Data Transforms
# ============================================================================

print("\n" + "="*60)
print("DEFINING TRANSFORMS")
print("="*60)

normalize = Normalize(mean=image_mean, std=image_std)

# Training transforms (with augmentation)
train_transforms = Compose([
    RandomResizedCrop(size, scale=(0.8, 1.0)),
    RandomHorizontalFlip(p=0.5),
    RandomRotation(10),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ToTensor(),
    normalize
])

# Validation/Test transforms (no augmentation)
val_transforms = Compose([
    Resize(size),
    CenterCrop(size),
    ToTensor(),
    normalize
])

print("✓ Training transforms: Augmentation enabled")
print("✓ Validation/Test transforms: No augmentation")

# ============================================================================
# CELL 10: Apply Transforms to Datasets
# ============================================================================

print("\n" + "="*60)
print("APPLYING TRANSFORMS")
print("="*60)

def apply_transforms(examples, transforms):
    """Apply image transforms"""
    examples["pixel_values"] = [
        transforms(img.convert("RGB")) for img in examples["image"]
    ]
    del examples["image"]
    return examples

print("Applying transforms to training set...")
train_ds_transformed = train_ds.map(
    lambda ex: apply_transforms(ex, train_transforms),
    batched=False,
    remove_columns=['image']
)

print("Applying transforms to validation set...")
val_ds_transformed = val_ds.map(
    lambda ex: apply_transforms(ex, val_transforms),
    batched=False,
    remove_columns=['image']
)

print("Applying transforms to test set...")
test_ds_transformed = test_ds.map(
    lambda ex: apply_transforms(ex, val_transforms),
    batched=False,
    remove_columns=['image']
)

print("✓ All transforms applied successfully!")

# ============================================================================
# CELL 11: Data Collator
# ============================================================================

print("\n" + "="*60)
print("SETTING UP DATA COLLATOR")
print("="*60)

def collate_fn(examples):
    """Batch processing function"""
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
    labels = torch.tensor([ex["label"] for ex in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# Test batch
test_batch = collate_fn(train_ds_transformed[:4])
print(f"✓ Batch shapes:")
print(f"  - pixel_values: {test_batch['pixel_values'].shape}")
print(f"  - labels: {test_batch['labels'].shape}")

# ============================================================================
# CELL 12: Load Vim Model from HuggingFace
# ============================================================================

print("\n" + "="*60)
print("LOADING VIM MODEL")
print("="*60)

print("Loading pretrained Vim-tiny from HuggingFace Hub...")
model = AutoModelForImageClassification.from_pretrained(
    model_name,
    num_labels=num_labels,  # 7 emotion classes
    id2label=id2label,
    label2id=label2id,
    trust_remote_code=True,  # Allow HF to download Vim's custom code
)

model.to(device)
print(f"✓ Model loaded successfully!")
print(f"✓ Pretrained backbone: ImageNet (1000 classes)")
print(f"✓ Fine-tune head: {num_labels} emotion classes")
print(f"✓ Model on device: {device}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✓ Total parameters: {total_params / 1e6:.1f}M")
print(f"✓ Trainable parameters: {trainable_params / 1e6:.1f}M")

# ============================================================================
# CELL 13: Training Arguments
# ============================================================================

print("\n" + "="*60)
print("TRAINING CONFIGURATION")
print("="*60)

output_dir = "/kaggle/working/vim-fer2013-model"
os.makedirs(output_dir, exist_ok=True)

train_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=15,  # Increased for better convergence
    learning_rate=5e-5,   # Conservative LR for pretrained model
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=3,
    logging_steps=50,
    remove_unused_columns=False,
    report_to="tensorboard",
    fp16=torch.cuda.is_available(),  # Mixed precision
    metric_for_best_model="accuracy",
    greater_is_better=True,
    warmup_steps=500,
    gradient_accumulation_steps=1,
    dataloader_pin_memory=False,
)

print(f"✓ Output directory: {output_dir}")
print(f"✓ Batch size: 32 (train) / 64 (eval)")
print(f"✓ Learning rate: {train_args.learning_rate}")
print(f"✓ Epochs: {train_args.num_train_epochs}")
print(f"✓ Mixed precision (FP16): {train_args.fp16}")
print(f"✓ Warmup steps: {train_args.warmup_steps}")

# ============================================================================
# CELL 14: Metrics Function
# ============================================================================

print("\n" + "="*60)
print("METRICS CONFIGURATION")
print("="*60)

def compute_metrics(eval_pred):
    """Compute metrics during evaluation"""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy}

print("✓ Metrics function defined: accuracy")

# ============================================================================
# CELL 15: Create Trainer
# ============================================================================

print("\n" + "="*60)
print("INITIALIZING TRAINER")
print("="*60)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_ds_transformed,
    eval_dataset=val_ds_transformed,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

print("✓ Trainer initialized!")
print(f"✓ Training samples: {len(train_ds_transformed)}")
print(f"✓ Validation samples: {len(val_ds_transformed)}")

# ============================================================================
# CELL 16: Train Model
# ============================================================================

print("\n" + "="*60)
print("🚀 STARTING TRAINING 🚀")
print("="*60)

try:
    train_results = trainer.train()
    print("\n✅ TRAINING COMPLETED SUCCESSFULLY!")
    print(f"✓ Final loss: {train_results.training_loss:.4f}")
except Exception as e:
    print(f"❌ Training failed with error: {e}")
    raise

# ============================================================================
# CELL 17: Save Model
# ============================================================================

print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

save_dir = "/kaggle/working/vim-fer2013-final"
os.makedirs(save_dir, exist_ok=True)

# Save using trainer (includes everything)
trainer.save_model(save_dir)
processor.save_pretrained(save_dir)

# Save config
config_dict = {
    "num_labels": num_labels,
    "id2label": id2label,
    "label2id": label2id,
    "model_name": model_name,
    "image_size": size,
    "image_mean": image_mean,
    "image_std": image_std,
}

with open(os.path.join(save_dir, "training_config.json"), "w") as f:
    json.dump(config_dict, f, indent=2)

print(f"✓ Model saved to: {save_dir}")
print(f"✓ Files:")
print(f"  - pytorch_model.bin (model weights)")
print(f"  - preprocessor_config.json (processor)")
print(f"  - training_config.json (configuration)")

# ============================================================================
# CELL 18: Evaluate on Test Set
# ============================================================================

print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

def evaluate_on_test(model, test_dataset, collate_fn, device, batch_size=64):
    """Evaluate model on test set"""
    model.eval()
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=False
    )
    
    all_logits = []
    all_labels = []
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(test_dataloader)}")
            
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(pixel_values=pixel_values)
            all_logits.append(outputs.logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    logits = np.concatenate(all_logits)
    y_true = np.concatenate(all_labels)
    y_pred = np.argmax(logits, axis=1)
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n✓ Test completed!")
    print(f"✓ Test samples evaluated: {len(y_true)}")
    
    return y_true, y_pred, logits

y_true, y_pred, logits = evaluate_on_test(model, test_ds_transformed, collate_fn, device)

# ============================================================================
# CELL 19: Classification Report
# ============================================================================

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60 + "\n")

report = classification_report(
    y_true, 
    y_pred, 
    target_names=labels, 
    digits=4,
    zero_division=0
)
print(report)

# Save report to file
with open("/kaggle/working/classification_report.txt", "w") as f:
    f.write("="*60 + "\n")
    f.write("VIM FINE-TUNING ON FER2013 - CLASSIFICATION REPORT\n")
    f.write("="*60 + "\n\n")
    f.write(report)

print("✓ Report saved to: /kaggle/working/classification_report.txt")

# ============================================================================
# CELL 20: Confusion Matrix
# ============================================================================

print("\n" + "="*60)
print("CONFUSION MATRIX")
print("="*60)

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot and save
fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(cm, display_labels=labels)
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title('FER2013 Confusion Matrix - Vim Fine-Tuned', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('/kaggle/working/confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✓ Confusion matrix saved to: /kaggle/working/confusion_matrix.png")
plt.show()

# ============================================================================
# CELL 21: Per-Class Accuracy & Metrics
# ============================================================================

print("\n" + "="*60)
print("PER-CLASS ACCURACY & METRICS")
print("="*60 + "\n")

per_class_acc = {}
for i, label in enumerate(labels):
    mask = y_true == i
    if mask.sum() > 0:
        acc = (y_pred[mask] == i).mean()
        per_class_acc[label] = {
            'accuracy': acc,
            'samples': mask.sum(),
            'correct': (y_pred[mask] == i).sum()
        }
        print(f"{label:15s}: {acc:.4f} ({per_class_acc[label]['correct']:3d}/{mask.sum():3d} samples)")

# Overall accuracy
overall_accuracy = (y_pred == y_true).mean()
print(f"\n{'Overall':15s}: {overall_accuracy:.4f}")

# ============================================================================
# CELL 22: Single Image Inference Function
# ============================================================================

print("\n" + "="*60)
print("INFERENCE FUNCTION")
print("="*60)

def predict_emotion(image_path, model, processor, device, id2label):
    """
    Predict emotion from a single image
    
    Args:
        image_path: Path to image file
        model: Trained model
        processor: Image processor
        device: torch device
        id2label: Label mapping
    
    Returns:
        emotion: Predicted emotion
        confidence: Confidence score
    """
    image = Image.open(image_path).convert('RGB')
    
    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        pred_id = probabilities.argmax(-1).item()
        confidence = probabilities[0, pred_id].item()
    
    emotion = id2label[pred_id]
    return emotion, confidence, probabilities[0].cpu().numpy()

print("✓ Inference function defined")
print("✓ Usage: emotion, conf, probs = predict_emotion(image_path, model, processor, device, id2label)")

# ============================================================================
# CELL 23: Test Inference on Sample
# ============================================================================

print("\n" + "="*60)
print("TESTING INFERENCE")
print("="*60)

# Get a random test sample
test_sample = test_ds_transformed[0]
test_image = test_sample['image'] if 'image' in test_sample else None

if test_image is None:
    # If image is already transformed, we need to get it from original dataset
    test_sample_orig = test_ds[0]
    test_image_pil = test_sample_orig['image']
    
    # Save temporarily for inference
    temp_path = '/tmp/test_sample.jpg'
    test_image_pil.save(temp_path)
    
    # Predict
    emotion, conf, probs = predict_emotion(temp_path, model, processor, device, id2label)
    true_label = labels[test_sample_orig['label']]
    
    print(f"\n✓ Test sample:")
    print(f"  True emotion: {true_label}")
    print(f"  Predicted emotion: {emotion}")
    print(f"  Confidence: {conf:.4f}")
    print(f"\n  All predictions:")
    for label, prob in zip(labels, probs):
        print(f"    {label:15s}: {prob:.4f}")

# ============================================================================
# CELL 24: Summary & Results
# ============================================================================

print("\n" + "="*70)
print(" "*15 + "🎉 VIM FINE-TUNING COMPLETE! 🎉")
print("="*70)

print("\n📊 FINAL RESULTS:")
print(f"  • Overall Test Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
print(f"  • Best per-class accuracy: {max([v['accuracy'] for v in per_class_acc.values()]):.4f}")
print(f"  • Worst per-class accuracy: {min([v['accuracy'] for v in per_class_acc.values()]):.4f}")

print("\n📁 OUTPUT FILES:")
print(f"  • Model: {save_dir}/")
print(f"  • Classification Report: /kaggle/working/classification_report.txt")
print(f"  • Confusion Matrix: /kaggle/working/confusion_matrix.png")
print(f"  • Sample Images: /kaggle/working/vim_samples.png")

print("\n💾 MODEL LOADING:")
print(f"  model = AutoModelForImageClassification.from_pretrained('{save_dir}')")
print(f"  processor = AutoImageProcessor.from_pretrained('{save_dir}')")

print("\n✅ Ready for deployment or further fine-tuning!")
print("="*70)

# ============================================================================
# END OF SCRIPT
# ============================================================================
