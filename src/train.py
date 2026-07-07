import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add root directory to python path for clean relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import HandwritingCNN
from src.dataset import get_emnist_dataloaders


def train_model(
    epochs: int = 5,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    model_save_path: str = "models/ocr_model.pth"
):
    """
    Orchestrates the entire training and validation pipeline for the custom CNN.
    """
    # 1. Device configuration (Train on GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Get DataLoaders from dataset module
    print("Loading datasets...")
    train_loader, val_loader, _, num_classes = get_emnist_dataloaders(
        data_dir="./data",
        batch_size=batch_size,
        val_split=0.1
    )

    # 3. Instantiate model, loss function, and optimizer
    model = HandwritingCNN(num_classes=num_classes).to(device)
    
    # CrossEntropyLoss: Combines Softmax activation and Negative Log-Likelihood loss
    # in a single class. Highly stable for multi-class classification (47 categories).
    criterion = nn.CrossEntropyLoss()
    
    # Adam: A popular optimizer that computes adaptive learning rates for each parameter,
    # leading to faster training convergence compared to standard SGD.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0
    start_time = time.time()

    print("\nStarting Training Loop...")
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        
        # ==========================================
        #              TRAINING PHASE
        # ==========================================
        # model.train(): Tells PyTorch that the model is in training mode.
        # This enables layers like Dropout and Batch Normalization to act appropriately.
        model.train()
        
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # tqdm wraps the loader to show a clean progress bar in terminal
        train_bar = tqdm(train_loader, desc="Training")
        for images, labels in train_bar:
            # Move data to the selected device (CPU or GPU)
            images, labels = images.to(device), labels.to(device)
            
            # Reset gradients: PyTorch accumulates gradients from previous runs by default.
            # We must clear them before calculating gradients for the new batch.
            optimizer.zero_grad()
            
            # A. FORWARD PASS: Pass images through the model to get raw logit scores
            outputs = model(images)
            
            # B. LOSS CALCULATION: Quantify how far off predictions are from ground truth labels
            loss = criterion(outputs, labels)
            
            # C. BACKPROPAGATION: Computes gradients of the loss function with respect to 
            # all learnable model weights using the chain rule.
            loss.backward()
            
            # D. OPTIMIZER STEP: Updates the model's weights using the computed gradients 
            # and the learning rate configuration.
            optimizer.step()
            
            # Track loss and batch accuracies
            running_train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # Update visual progress bar stats
            train_bar.set_postfix(
                loss=f"{loss.item():.4f}", 
                acc=f"{100.0 * correct_train / total_train:.2f}%"
            )
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_acc = 100.0 * correct_train / total_train

        # ==========================================
        #             VALIDATION PHASE
        # ==========================================
        # model.eval(): Tells PyTorch the model is in evaluation mode.
        # This disables Dropout (keeps all neurons active) and uses running stats for BatchNorm.
        model.eval()
        
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        # torch.no_grad(): Disables the autograd engine (gradient tracking).
        # We do not compute gradients during validation, saving memory and CPU/GPU cycles.
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass (no backpropagation needed)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = 100.0 * correct_val / total_val

        # Print Epoch Summary Metrics
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.2f}%")

        # Save model if validation accuracy improves
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            # Save the model's learned weights (parameters)
            torch.save(model.state_dict(), model_save_path)
            print(f"==> Saved new best model checkpoint to {model_save_path}!")

    total_time = time.time() - start_time
    print("\n" + "=" * 40)
    print("=== Training Completion Summary ===")
    print("=" * 40)
    print(f"Total training time: {total_time / 60:.2f} minutes")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved at: {model_save_path}")


if __name__ == "__main__":
    # Standard settings for a quick training run verification
    # We set epochs=1 to verify functionality first
    train_model(
        epochs=1,
        batch_size=128,
        learning_rate=0.001,
        model_save_path="models/ocr_model.pth"
    )
