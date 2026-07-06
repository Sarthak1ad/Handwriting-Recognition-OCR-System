import cv2
import numpy as np


def preprocess_character(char_img: np.ndarray, target_size: int = 28, padding: int = 4) -> np.ndarray:
    """
    Pads the character to a square canvas preserving aspect ratio,
    then resizes it to target_size x target_size.
    
    This matches the format of the EMNIST training dataset (28x28, centered).
    """
    h, w = char_img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_size, target_size), dtype=np.uint8)
        
    # Determine the larger dimension to create a square canvas
    max_dim = max(h, w)
    
    # Calculate padding for centering
    pad_h = (max_dim - h) // 2 + padding
    pad_w = (max_dim - w) // 2 + padding
    
    # Add border around the character image (black background, value 0)
    padded = cv2.copyMakeBorder(
        char_img,
        top=pad_h,
        bottom=pad_h + (1 if (max_dim - h) % 2 != 0 else 0),
        left=pad_w,
        right=pad_w + (1 if (max_dim - w) % 2 != 0 else 0),
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )
    
    # Resize to target canvas size (28x28)
    resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized


def segment_lines(binary_image: np.ndarray) -> list[tuple[int, int]]:
    """
    Finds horizontal segments containing text lines using horizontal projection profiles.
    Returns a list of (y_start, y_end) tuples.
    """
    row_sums = np.sum(binary_image, axis=1)
    
    # Define a small threshold to ignore minor noise (0.5% of max possible row sum)
    threshold = binary_image.shape[1] * 255 * 0.005
    
    in_line = False
    y_start = 0
    lines = []
    
    for y in range(len(row_sums)):
        if row_sums[y] > threshold and not in_line:
            y_start = y
            in_line = True
        elif row_sums[y] <= threshold and in_line:
            y_end = y
            # Ignore lines that are too small (less than 5 pixels in height)
            if y_end - y_start >= 5:
                lines.append((y_start, y_end))
            in_line = False
            
    if in_line:
        if binary_image.shape[0] - y_start >= 5:
            lines.append((y_start, binary_image.shape[0]))
            
    return lines


def segment_words(line_image: np.ndarray, y_offset: int) -> list[tuple[int, int, int, int]]:
    """
    Finds bounding boxes for words within a line using horizontal dilation 
    followed by contour detection.
    
    Returns:
        List of absolute bounding boxes (x, y, w, h) sorted from left to right.
    """
    # Dilate horizontally to connect characters in the same word
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    dilated = cv2.dilate(line_image, kernel, iterations=1)
    
    # Find contours on dilated line
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    word_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 3 and h > 3:
            word_boxes.append((x, y_offset + y, w, h))
            
    # Sort words left-to-right
    word_boxes.sort(key=lambda box: box[0])
    return word_boxes


def segment_characters(word_image: np.ndarray, word_box: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    """
    Extracts individual character bounding boxes within a word image using contours.
    
    Returns:
        List of absolute bounding boxes (x, y, w, h) sorted from left to right.
    """
    contours, _ = cv2.findContours(word_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    char_boxes = []
    wx, wy, ww, wh = word_box
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= 2 and h >= 2:
            char_boxes.append((wx + x, wy + y, w, h))
            
    # Sort characters left-to-right
    char_boxes.sort(key=lambda box: box[0])
    return char_boxes


def segment_image(binary_image: np.ndarray) -> list[dict]:
    """
    Performs full hierarchical segmentation on a binary image.
    
    Returns:
        A list of dictionaries representing the line/word/character structure.
        Each character entry includes its bounding box and 28x28 normalized image.
    """
    segmented_data = []
    lines = segment_lines(binary_image)
    
    for y_start, y_end in lines:
        line_box = (0, y_start, binary_image.shape[1], y_end - y_start)
        line_img = binary_image[y_start:y_end, :]
        
        words = []
        word_boxes = segment_words(line_img, y_start)
        
        for wx, wy, ww, wh in word_boxes:
            word_img = binary_image[wy:wy+wh, wx:wx+ww]
            characters = []
            char_boxes = segment_characters(word_img, (wx, wy, ww, wh))
            
            for cx, cy, cw, ch in char_boxes:
                char_img = binary_image[cy:cy+ch, cx:cx+cw]
                processed_char = preprocess_character(char_img)
                
                characters.append({
                    "box": (cx, cy, cw, ch),
                    "image": processed_char
                })
                
            if characters:
                words.append({
                    "box": (wx, wy, ww, wh),
                    "characters": characters
                })
                
        if words:
            segmented_data.append({
                "box": line_box,
                "words": words
            })
            
    return segmented_data
