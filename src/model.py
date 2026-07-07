import torch
import torch.nn as nn


class HandwritingCNN(nn.Module):
    """
    A simple, learning-focused Convolutional Neural Network (CNN) for Handwriting OCR,
    trained on the EMNIST Balanced dataset (47 classes).
    
    Architecture:
    Input (1, 28, 28) -> Conv1 -> ReLU -> MaxPool1 -> Conv2 -> ReLU -> MaxPool2 -> Flatten -> FC1 -> ReLU -> Dropout -> FC2 (47)
    """
    def __init__(self, num_classes: int = 47):
        super(HandwritingCNN, self).__init__()
        
        # 1. Feature Extractor (Convolutional Blocks)
        # CONV2D: We use Conv2d because images have spatial structures (height, width).
        # A 2D convolution slides filters over the image to detect local features 
        # like edges, curves, and junctions of lines in handwriting.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # RELU: The Rectified Linear Unit (ReLU) introduces non-linearity.
        # Without non-linear activation functions, a multi-layer network behaves 
        # just like a single linear model, unable to learn complex handwriting patterns.
        self.relu = nn.ReLU()
        
        # MAXPOOL2D: Downsamples spatial dimensions (height and width) by taking 
        # the maximum value in a window (2x2). This reduces computational complexity, 
        # memory footprint, and provides translation invariance (so a character is 
        # recognized even if it shifts slightly).
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2. Classifier (Fully Connected Blocks)
        # FLATTEN: Converts the 2D feature maps (channels, height, width) into a 1D vector.
        # Linear layers expect a flat 1D vector of inputs per sample, not a 3D grid.
        # Input shape to Flatten: (64, 7, 7) -> Output shape: (3136,)
        self.flatten = nn.Flatten()
        
        # LINEAR (Fully Connected): Connects every input to every output.
        # Takes the high-level extracted features and combines them to make predictions.
        # FC1 maps the 3136 features to a dense representation of 128 features.
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        
        # DROPOUT: A regularization technique. During training, it randomly sets 
        # 50% of the activations to 0. This prevents the network from relying too much 
        # on specific neurons, preventing overfitting and encouraging robust features.
        self.dropout = nn.Dropout(p=0.5)
        
        # FC2 maps 128 features to the final 47 classes (scores for each EMNIST class).
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass execution of the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Output logit tensor of shape (batch_size, 47)
        """
        # --- Block 1 ---
        # Input: [B, 1, 28, 28]
        x = self.conv1(x)  # Output: [B, 32, 28, 28] (32 filters detect simple shapes)
        x = self.relu(x)   # Output: [B, 32, 28, 28] (Adds non-linearity)
        x = self.pool(x)   # Output: [B, 32, 14, 14] (Reduces size by half)
        
        # --- Block 2 ---
        x = self.conv2(x)  # Output: [B, 64, 14, 14] (64 filters detect complex shapes)
        x = self.relu(x)   # Output: [B, 64, 14, 14] (Adds non-linearity)
        x = self.pool(x)   # Output: [B, 64, 7, 7]   (Reduces size by half)
        
        # --- Classification Head ---
        x = self.flatten(x)  # Output: [B, 3136] (Flattens 64 channels of 7x7 maps into 3136 elements)
        x = self.fc1(x)      # Output: [B, 128]  (Dense representation)
        x = self.relu(x)     # Output: [B, 128]  (Adds non-linearity)
        x = self.dropout(x)  # Output: [B, 128]  (Randomly zeroes 50% of values for regularization)
        x = self.fc2(x)      # Output: [B, 47]   (Logit scores for the 47 classes)
        
        return x


if __name__ == "__main__":
    print("=== Handwriting CNN Model Verification ===")
    
    # 1. Instantiate the model
    model = HandwritingCNN(num_classes=47)
    print("\nModel Architecture:")
    print(model)
    
    # 2. Generate a dummy tensor representing a single EMNIST image
    # Shape: (batch_size=1, channels=1, height=28, width=28)
    dummy_input = torch.randn(1, 1, 28, 28)
    print(f"\nInput Tensor Shape: {dummy_input.shape} (Expected: [1, 1, 28, 28])")
    
    # 3. Perform a forward pass
    # Set model to evaluation mode (disables dropout)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        
    print(f"Output Tensor Shape: {output.shape} (Expected: [1, 47])")
    print(f"Number of target classes: {output.shape[1]}")
    
    # Verify that forward pass runs successfully and output is numeric
    assert output.shape == (1, 47), "Error: Output shape does not match target classes!"
    print("\nCNN Model designed and verified successfully!")
