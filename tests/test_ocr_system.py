import os
import sys
import pytest
import numpy as np
import torch
import cv2

# Add root directory to python search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import convert_to_grayscale, apply_blur, binarize_image
from src.segmentation import preprocess_character, segment_lines
from src.dataset import get_emnist_transforms, EMNIST_CLASSES
from src.model import HandwritingCNN
from src.inference import prepare_image, load_trained_model
from src.ocr_pipeline import run_ocr_pipeline


# ==========================================
#          PREPROCESSING TESTS
# ==========================================

def test_convert_to_grayscale():
    """Test that grayscale conversion maps 3 channels to 1 channel."""
    # Create random BGR image
    dummy_color = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    gray = convert_to_grayscale(dummy_color)
    assert len(gray.shape) == 2
    assert gray.shape == (100, 100)
    
    # Passing an already grayscale image should return it as is
    gray_same = convert_to_grayscale(gray)
    assert gray_same.shape == (100, 100)


def test_apply_blur():
    """Test that Gaussian blur maintains image dimensions."""
    dummy_gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    blurred = apply_blur(dummy_gray, kernel_size=5)
    assert blurred.shape == (100, 100)


def test_binarize_image():
    """Test binarization outputs are binary (only 0 or 255 values)."""
    # Create gradient grayscale image
    dummy_gray = np.linspace(0, 255, 10000, dtype=np.uint8).reshape(100, 100)
    binary = binarize_image(dummy_gray)
    assert binary.shape == (100, 100)
    # Check that pixels are strictly binary
    unique_vals = np.unique(binary)
    for val in unique_vals:
        assert val in [0, 255]


# ==========================================
#          SEGMENTATION TESTS
# ==========================================

def test_preprocess_character():
    """Test that arbitrary shapes are resized and padded to 28x28."""
    # Narrow high rectangle (e.g. '1' or 'l')
    narrow_char = np.ones((50, 15), dtype=np.uint8) * 255
    resized = preprocess_character(narrow_char, target_size=28, padding=4)
    assert resized.shape == (28, 28)
    
    # Wide flat rectangle
    wide_char = np.ones((10, 40), dtype=np.uint8) * 255
    resized_wide = preprocess_character(wide_char, target_size=28, padding=4)
    assert resized_wide.shape == (28, 28)


def test_segment_lines():
    """Test that line segmentation finds horizontal gaps."""
    # Create canvas with two horizontal white stripes (text) on black background
    canvas = np.zeros((100, 200), dtype=np.uint8)
    canvas[20:30, :] = 255
    canvas[60:75, :] = 255
    
    lines = segment_lines(canvas)
    assert len(lines) == 2
    # Verify approximate coordinates of lines
    assert abs(lines[0][0] - 20) <= 2
    assert abs(lines[0][1] - 30) <= 2
    assert abs(lines[1][0] - 60) <= 2
    assert abs(lines[1][1] - 75) <= 2


# ==========================================
#            DATASET TESTS
# ==========================================

def test_dataset_transforms():
    """Verify that dataset transforms are instantiated successfully."""
    transform = get_emnist_transforms()
    assert transform is not None


def test_classes_count():
    """Assert EMNIST Balanced class mappings have 47 categories."""
    assert len(EMNIST_CLASSES) == 47
    assert '0' in EMNIST_CLASSES
    assert 'Z' in EMNIST_CLASSES
    assert 't' in EMNIST_CLASSES


# ==========================================
#             MODEL TESTS
# ==========================================

def test_model_forward():
    """Verify that the model maps inputs of shape (B, 1, 28, 28) to output score logits (B, 47)."""
    model = HandwritingCNN(num_classes=47)
    
    # Test batch size of 2
    dummy_input = torch.randn(2, 1, 28, 28)
    outputs = model(dummy_input)
    assert outputs.shape == (2, 47)


# ==========================================
#           INFERENCE TESTS
# ==========================================

def test_prepare_image():
    """Verify inference image preparations yields standard (1, 1, 28, 28) tensors."""
    # Test passing a grayscale 64x64 NumPy array
    dummy_img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    tensor = prepare_image(dummy_img)
    assert tensor.shape == (1, 1, 28, 28)


# ==========================================
#          OCR PIPELINE TESTS
# ==========================================

def test_ocr_pipeline_dry_run():
    """Run a dry run of the complete OCR pipeline on a simple mock canvas."""
    # Create black canvas with a simple drawn character
    canvas = np.ones((100, 300, 3), dtype=np.uint8) * 255
    cv2.putText(canvas, "A B", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    model_path = "models/ocr_model.pth"
    if os.path.exists(model_path):
        text, metadata = run_ocr_pipeline(canvas, model_path=model_path, save_vis=False)
        # Should return reconstructed string
        assert isinstance(text, str)
        # Metadata list should contain predictions for detected letters
        assert len(metadata) > 0
        for item in metadata:
            assert "box" in item
            assert "character" in item
            assert "confidence" in item
    else:
        # If no model is trained yet, skip this specific integration assert
        pass
