import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

# Ensure the project root is available for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import EMNIST_CLASSES
from src.model import HandwritingCNN
from src.preprocessing import preprocess_image
from src.segmentation import preprocess_character


def load_trained_model(model_path: str = "models/ocr_model.pth") -> tuple[torch.nn.Module, torch.device]:
    """Load the saved CNN weights and prepare the model for inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = HandwritingCNN(num_classes=47)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # model.eval() switches off training-only behavior such as Dropout.
    # This makes predictions consistent and deterministic during inference.
    model.eval()
    return model, device


def prepare_image(image_input: str | np.ndarray) -> torch.Tensor:
    """Convert an image path or array into the same tensor format used in training."""
    if isinstance(image_input, (str, os.PathLike)):
        image = cv2.imread(str(image_input), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {image_input}")
    else:
        image = np.array(image_input)
        if image.ndim == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                image = image[:, :, 0]

    if image.shape != (28, 28):
        # Reuse the preprocessing flow from preprocessing.py for raw inputs.
        if image.ndim == 2 and image.shape[0] > 32 and image.shape[1] > 32:
            processed = preprocess_image(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), deskew=True)
            image = cv2.resize(processed, (28, 28), interpolation=cv2.INTER_AREA)
        else:
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            image = preprocess_character(binary)

    # Match the training transform as closely as possible: grayscale -> transpose -> tensor -> normalize.
    pil_image = Image.fromarray(image.astype(np.uint8), mode="L")
    pil_image = pil_image.transpose(Image.TRANSPOSE)
    tensor = torch.tensor(np.array(pil_image, dtype=np.float32) / 255.0)
    tensor = (tensor - 0.1736) / 0.3317
    return tensor.unsqueeze(0).unsqueeze(0)


def predict_character(image_path: str, model_path: str = "models/ocr_model.pth") -> dict:
    """Return the predicted class index, character, and confidence score."""
    model, device = load_trained_model(model_path)
    tensor = prepare_image(image_path).to(device)

    # torch.no_grad() disables gradient tracking because inference does not need backpropagation.
    # It saves memory and makes prediction faster on CPU or GPU.
    with torch.no_grad():
        logits = model(tensor)

        # Softmax converts raw model scores into probabilities that sum to 1.0.
        # This makes the output easier to interpret as a confidence score.
        probabilities = torch.softmax(logits, dim=1)

        # The highest probability is selected because it is the most likely class.
        confidence, class_index = torch.max(probabilities, dim=1)

    confidence_pct = round(confidence.item() * 100.0, 2)
    return {
        "class_index": class_index.item(),
        "character": EMNIST_CLASSES[class_index.item()],
        "confidence": confidence_pct,
    }


if __name__ == "__main__":
    print("=== Handwriting CNN Inference Verification ===")

    sample_dir = os.path.join("data", "output")
    os.makedirs(sample_dir, exist_ok=True)

    sample_path = os.path.join(sample_dir, "sample_inference.png")
    if not os.path.exists(sample_path):
        canvas = np.zeros((64, 64), dtype=np.uint8)
        cv2.putText(canvas, "A", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 2)
        cv2.imwrite(sample_path, canvas)

    print(f"Loading sample image: {sample_path}")
    result = predict_character(sample_path)

    print("\n--- Prediction Results ---")
    print(f"Predicted Class Index: {result['class_index']}")
    print(f"Predicted Character:   {result['character']}")
    print(f"Confidence Score:      {result['confidence']:.2f}%")
    print("\nInference pipeline verified successfully!")
