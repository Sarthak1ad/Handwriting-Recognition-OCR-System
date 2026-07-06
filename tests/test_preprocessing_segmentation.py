import os
import sys
import cv2
import numpy as np

# Ensure root directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import preprocess_image
from src.segmentation import segment_image


def generate_synthetic_skewed_text() -> np.ndarray:
    """
    Creates a synthetic BGR image containing text on two lines, 
    skewed (rotated) by a small angle to test deskewing.
    """
    # Create a white canvas (background = 255)
    canvas = np.ones((200, 800, 3), dtype=np.uint8) * 255
    
    # Draw two lines of text in black (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "HELLO", (100, 70), font, 1.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, "WORLD", (450, 70), font, 1.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, "TEST 123", (100, 150), font, 1.8, (0, 0, 0), 3, cv2.LINE_AA)
    
    # Rotate canvas by 3.5 degrees to skew it
    h, w = canvas.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 3.5, 1.0)
    
    # Warp image with white border fill
    skewed_image = cv2.warpAffine(
        canvas,
        rotation_matrix,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    return skewed_image


def test_pipeline():
    """
    Integrative test checking preprocessing and hierarchical segmentation.
    Saves visual results in the data/ directory.
    """
    print("Generating skewed synthetic text image...")
    img = generate_synthetic_skewed_text()
    
    # Setup data output directory
    output_dir = os.path.join("data", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save input image
    input_path = os.path.join(output_dir, "0_input_skewed.png")
    cv2.imwrite(input_path, img)
    print(f"Saved skewed synthetic input to {input_path}")
    
    # 1. Run Preprocessing (binarizes, inverts, and deskews)
    print("\nRunning Preprocessing Pipeline...")
    binary_img = preprocess_image(img, deskew=True)
    
    # Save binary image
    binary_path = os.path.join(output_dir, "1_preprocessed_binary.png")
    cv2.imwrite(binary_path, binary_img)
    print(f"Saved preprocessed binary mask to {binary_path}")
    
    # 2. Run Segmentation
    print("\nRunning Hierarchical Segmentation...")
    segmented_data = segment_image(binary_img)
    
    # We expect 2 lines (Line 1: "HELLO" "WORLD", Line 2: "TEST 123")
    print(f"Detected {len(segmented_data)} text lines.")
    assert len(segmented_data) > 0, "Failed to detect any lines."
    
    # Prepare a visualization canvas from the preprocessed binary (converted back to BGR for drawing colored boxes)
    vis_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    
    char_count = 0
    for line_idx, line in enumerate(segmented_data):
        # Draw Line Bounding Box (Blue)
        lx, ly, lw, lh = line["box"]
        cv2.rectangle(vis_img, (lx, ly), (lx + lw, ly + lh), (255, 0, 0), 2)
        print(f"  Line {line_idx+1}: Box={line['box']}, containing {len(line['words'])} words.")
        
        for word_idx, word in enumerate(line["words"]):
            # Draw Word Bounding Box (Green)
            wx, wy, ww, wh = word["box"]
            cv2.rectangle(vis_img, (wx, wy), (wx + ww, wy + wh), (0, 255, 0), 2)
            
            for char_idx, char in enumerate(word["characters"]):
                # Draw Character Bounding Box (Red)
                cx, cy, cw, ch = char["box"]
                cv2.rectangle(vis_img, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 1)
                
                # Check character shape assertion
                char_img = char["image"]
                assert char_img.shape == (28, 28), f"Incorrect character shape: {char_img.shape}"
                
                # Save each individual normalized character
                char_count += 1
                char_filename = f"char_{line_idx+1}_{word_idx+1}_{char_idx+1}.png"
                cv2.imwrite(os.path.join(output_dir, char_filename), char_img)
                
    # Save the visualization of bounding boxes
    vis_path = os.path.join(output_dir, "2_segmentation_boxes.png")
    cv2.imwrite(vis_path, vis_img)
    print(f"\nSaved segmentation boxes visualization to {vis_path}")
    print(f"Successfully processed and verified {char_count} individual characters.")
    print("All preprocessing & segmentation assertions passed successfully!")


if __name__ == "__main__":
    test_pipeline()
