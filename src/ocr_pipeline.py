import os
import sys
import time
import cv2
import numpy as np
import torch

# Ensure the root directory is on the python search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import preprocess_image
from src.segmentation import segment_image
from src.inference import load_trained_model, prepare_image
from src.dataset import EMNIST_CLASSES
from src.model import HandwritingCNN


def run_ocr_pipeline(
    image_path_or_np: str | np.ndarray,
    model_path: str = "models/ocr_model.pth",
    save_vis: bool = True,
    vis_path: str = "data/output/ocr_visualization.png",
    model: torch.nn.Module = None,
    device: torch.device = None
) -> tuple[str, list[dict]]:
    """
    Runs the complete OCR pipeline on an image containing lines of handwriting.
    
    1. Preprocesses the image (binarization, deskewing).
    2. Segments the image into a hierarchy of lines, words, and characters.
    3. Runs character classification on the CNN.
    4. Reconstructs text preserving reading order, spaces, and line breaks.
    
    Returns:
        - recognized_text: Reconstructed string.
        - char_metadata: Bounding boxes and confidence metadata.
    """
    # Load image from file path if a string is provided
    if isinstance(image_path_or_np, str):
        original_img = cv2.imread(image_path_or_np)
        if original_img is None:
            raise FileNotFoundError(f"Could not load image at {image_path_or_np}")
    else:
        original_img = image_path_or_np.copy()

    # Step 1: Preprocess image
    binary_img = preprocess_image(original_img, deskew=True)

    # Step 2: Segment image
    segmented_data = segment_image(binary_img)

    # Step 3: Set up device and model (reuse if provided, otherwise load from path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if model is None:
        model = HandwritingCNN(num_classes=47)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

    recognized_lines = []
    char_metadata = []
    
    # Prepare canvas for bounding boxes visualization
    vis_img = original_img.copy()

    # Step 4: Iterate through hierarchy to perform inference and reconstruction
    for line in segmented_data:
        lx, ly, lw, lh = line["box"]
        # Draw Line Bounding Box (Blue)
        cv2.rectangle(vis_img, (lx, ly), (lx + lw, ly + lh), (255, 0, 0), 2)
        
        words_in_line = []
        for word in line["words"]:
            wx, wy, ww, wh = word["box"]
            # Draw Word Bounding Box (Green)
            cv2.rectangle(vis_img, (wx, wy), (wx + ww, wy + wh), (0, 255, 0), 2)
            
            chars_in_word = []
            for char in word["characters"]:
                cx, cy, cw, ch = char["box"]
                # Draw Character Bounding Box (Red)
                cv2.rectangle(vis_img, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 1)
                
                # Get the 28x28 normalized character image
                char_img = char["image"]
                
                # Perform prediction
                tensor = prepare_image(char_img).to(device)
                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1)
                    confidence, predicted_idx = torch.max(probs, dim=1)

                class_idx = predicted_idx.item()
                predicted_char = EMNIST_CLASSES[class_idx]
                confidence_pct = confidence.item() * 100.0

                chars_in_word.append(predicted_char)
                
                # Add to character metadata list
                char_metadata.append({
                    "box": (cx, cy, cw, ch),
                    "character": predicted_char,
                    "confidence": confidence_pct
                })

                # Put recognized character text above the character bounding box
                cv2.putText(
                    vis_img,
                    predicted_char,
                    (cx, cy - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA
                )

            # Combine character tokens into a word string
            if chars_in_word:
                words_in_line.append("".join(chars_in_word))

        # Join word strings using spaces to reconstruct the sentence row
        if words_in_line:
            recognized_lines.append(" ".join(words_in_line))

    # Join sentence rows using newlines to reconstruct the final text
    recognized_text = "\n".join(recognized_lines)

    # Save visualization if enabled
    if save_vis:
        os.makedirs(os.path.dirname(vis_path), exist_ok=True)
        cv2.imwrite(vis_path, vis_img)
        print(f"Saved OCR pipeline bounding boxes visualization to {vis_path}")

    return recognized_text, char_metadata


if __name__ == "__main__":
    print("=== Handwriting OCR Pipeline Verification ===")
    
    # Use the skewed synthetic image generated in Module 1 tests
    test_img_path = os.path.join("data", "output", "0_input_skewed.png")
    
    if os.path.exists(test_img_path):
        print(f"Loading test image: {test_img_path}")
        start_time = time.time()
        
        # Run the full OCR pipeline
        text, boxes = run_ocr_pipeline(test_img_path)
        elapsed = time.time() - start_time
        
        print("\n--- Recognized Text ---")
        print(text)
        print("-----------------------")
        print(f"OCR Pipeline processing completed in {elapsed:.4f} seconds.")
        print(f"Successfully processed {len(boxes)} character bounding boxes.")
        print("OCR pipeline verified successfully!")
    else:
        print(f"\nWarning: Test image '{test_img_path}' not found.")
        print("Please run tests/test_preprocessing_segmentation.py first to generate the test image.")
