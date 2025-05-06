from transformers import AutoProcessor, AutoModelForVision2Seq # need pip install transformers 
from PIL import Image # need pip install torch
import torch # need pip install torch

#Another model_name : "Salesforce/blip-image-captioning-base"

class ImageCaptioner:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-large", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForVision2Seq.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("Note: This class takes an image path as input and generates a caption. It is not designed for question answering.")

    def write_caption(self, image_path: str, max_length: int = 100, num_beams: int = 3, early_stopping: bool = True) -> str:
        #check type
        self.__enforce_type(image_path, str, "image_path")
        self.__enforce_type(max_length, int, "max_length")
        self.__enforce_type(num_beams, int, "num_beams")
        self.__enforce_type(early_stopping, bool, "early_stopping")
        
        #check num_beam >= 2
        if num_beams < 2:
            raise ValueError(f"num_beams must be >= 2, but got {num_beams}") #Added the value to the error
        
        # load image
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found: {image_path}")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_length, num_beams=num_beams, early_stopping=early_stopping)

        # text return
        return self.processor.decode(outputs[0], skip_special_tokens=True)

    def __enforce_type(self, value, expected_type, arg_name):
        if not isinstance(value, expected_type):
            raise TypeError(f"Argument '{arg_name}' must be of type {expected_type.__name__}, but received {type(value).__name__}")

#Example
if __name__ == "__main__":
    _cap = ImageCaptioner()
    _text = _cap.write_caption("img.png")
    print("Caption:", _text)

