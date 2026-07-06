import cv2
import numpy as np


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converts a BGR or RGB image to grayscale.
    If the image is already grayscale, it returns it as is.
    """
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Applies Gaussian Blur to reduce noise and smooth the image.
    kernel_size must be an odd positive integer.
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def binarize_image(image: np.ndarray) -> np.ndarray:
    """
    Converts a grayscale image into a binary image using Otsu's thresholding.
    
    Since handwritten documents are typically dark ink on light paper,
    we use cv2.THRESH_BINARY_INV to make the text white (255) and the 
    background black (0), matching EMNIST dataset conventions.
    """
    _, binary = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary


def deskew_image(binary_image: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Corrects skew (tilt angle) of the text in the binary image.
    It calculates the minimum bounding rectangle enclosing all white pixels 
    and rotates the image to align it horizontally.
    
    Returns:
        - rotated_image: The deskewed binary image.
        - angle: The detected skew angle in degrees.
    """
    # Find coordinates of all white pixels (text region)
    coords = np.column_stack(np.where(binary_image > 0))
    if len(coords) == 0:
        return binary_image, 0.0

    # Find the minimum area rectangle enclosing these coordinates
    # cv2.minAreaRect returns: (center(x,y), size(w,h), angle of rotation)
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # In OpenCV, the angle returned depends on the version:
    # Older versions return angles in [-90, 0)
    # Newer versions return angles in [0, 90]
    # We standardize to rotate the tilt back to horizontal.
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle
    else:
        angle = -angle

    # Perform the rotation if the angle is significant
    if abs(angle) > 0.5:
        (h, w) = binary_image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(
            binary_image,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        return rotated_image, angle

    return binary_image, 0.0


def preprocess_image(image: np.ndarray, deskew: bool = True) -> np.ndarray:
    """
    Applies the full preprocessing pipeline:
    1. Grayscale conversion
    2. Gaussian blur for noise reduction
    3. Binarization (Otsu's inverted thresholding)
    4. Deskewing (rotation correction)
    
    Args:
        image: Original input BGR/RGB image.
        deskew: Whether to perform rotation correction.
        
    Returns:
        A preprocessed binary image ready for segmentation.
    """
    gray = convert_to_grayscale(image)
    blurred = apply_blur(gray, kernel_size=5)
    binary = binarize_image(blurred)
    
    if deskew:
        binary, _ = deskew_image(binary)
        
    return binary
