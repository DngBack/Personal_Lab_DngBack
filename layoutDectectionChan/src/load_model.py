"""
Target of this file is to load the model with input is file with transformer
"""
from transformers import AutoModelForImageTextToText, AutoProcessor

def load_model(model_path: str, device: str = "cuda") -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    """
    Load the model with input is file with transformer

    Args:
        model_path: str
        device: str = "cuda"
    Returns:
        tuple[AutoModelForImageTextToText, AutoProcessor]
    """
    model = AutoModelForImageTextToText.from_pretrained(model_path, device_map=device)
    processor = AutoProcessor.from_pretrained(model_path, device_map=device)
    return model, processor

