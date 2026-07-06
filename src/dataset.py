import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image

# EMNIST Balanced dataset has 47 classes representing digits and letters.
# The split merges similar-looking uppercase and lowercase characters (e.g., C/c, O/o)
# to make classification more robust.
EMNIST_CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]


def get_emnist_transforms() -> transforms.Compose:
    """
    Defines the preprocessing transforms for EMNIST characters.
    
    1. Transpose: EMNIST raw images are stored transposed (flipped diagonally).
       We transpose them back using PIL.Image.TRANSPOSE so they align with 
       real-world scanned/segmented characters.
    2. ToTensor: Converts PIL Image to PyTorch Tensor [0.0, 1.0] and swaps shape to (C, H, W).
    3. Normalize: Normalizes the tensor using EMNIST Balanced mean and standard deviation.
    """
    return transforms.Compose([
        transforms.Lambda(lambda img: img.transpose(Image.TRANSPOSE)),
        transforms.ToTensor(),
        # EMNIST Balanced standard stats: mean ~ 0.1736, std ~ 0.3317
        transforms.Normalize((0.1736,), (0.3317,))
    ])


def get_emnist_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 0
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Downloads EMNIST Balanced and returns Train, Validation, and Test DataLoaders.
    
    Args:
        data_dir: Directory where EMNIST dataset will be stored.
        batch_size: Number of samples per batch.
        val_split: Fraction of training dataset to use for validation.
        num_workers: Number of subprocesses to use for data loading.
        
    Returns:
        train_loader, val_loader, test_loader, and number of classes.
    """
    transform = get_emnist_transforms()
    
    # 1. Download and load full train dataset and test dataset
    full_train_dataset = datasets.EMNIST(
        root=data_dir,
        split="balanced",
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.EMNIST(
        root=data_dir,
        split="balanced",
        train=False,
        download=True,
        transform=transform
    )
    
    # 2. Split train dataset into train and validation sets
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    
    # Set seed for reproducible split
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size], 
        generator=generator
    )
    
    # 3. Create PyTorch DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    num_classes = len(EMNIST_CLASSES)
    
    return train_loader, val_loader, test_loader, num_classes


if __name__ == "__main__":
    print("=== EMNIST Balanced Dataset Verification ===")
    
    # Define parameters
    DATA_DIR = "./data"
    BATCH_SIZE = 128
    
    # Retrieve loaders
    train_loader, val_loader, test_loader, num_classes = get_emnist_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        val_split=0.1
    )
    
    # Print basic info
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {' '.join(EMNIST_CLASSES)}")
    print(f"Training dataset size: {len(train_loader.dataset)} samples")
    print(f"Validation dataset size: {len(val_loader.dataset)} samples")
    print(f"Testing dataset size: {len(test_loader.dataset)} samples")
    
    # Retrieve one batch to verify
    images, labels = next(iter(train_loader))
    
    print("\n--- Batch Verification ---")
    print(f"Image batch shape: {images.shape} (Expected: [batch_size, 1, 28, 28])")
    print(f"Label batch shape: {labels.shape} (Expected: [batch_size])")
    
    # Verification of values
    print(f"Image tensor min/max: {images.min().item():.4f} / {images.max().item():.4f}")
    print(f"Image tensor mean/std: {images.mean().item():.4f} / {images.std().item():.4f}")
    
    # Convert first few labels in the batch to characters
    sample_labels = labels[:15].tolist()
    sample_chars = [EMNIST_CLASSES[lbl] for lbl in sample_labels]
    print(f"First 15 labels in batch (numeric): {sample_labels}")
    print(f"First 15 labels in batch (character symbols): {sample_chars}")
    print("\nDataset preparation completed successfully!")
