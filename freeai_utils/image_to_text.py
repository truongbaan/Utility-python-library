from transformers import AutoProcessor, AutoModelForVision2Seq # need pip install transformers 
from PIL import Image # need pip install torch
import torch # need pip install torch
import logging
from typing import Optional
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

#Another model_name : "Salesforce/blip-image-captioning-base"

class ImageCaptioner:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-large", device: Optional[str] = None):
        #check input type
        self.__enforce_type(model_name, str, "model_name")
        
        #logger 
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Note: This class takes an image path as input and generates a caption. It is not designed for question answering.")
        
        #init the var to hold device available
        preferred_devices = []
        
        #try input first
        if device is not None:
            preferred_devices.append(device)
        
        # try cuda second 
        if torch.cuda.is_available() and "cuda" not in preferred_devices:
            preferred_devices.append("cuda")

        # fall back to CPU if not already there
        if "cpu" not in preferred_devices:
            preferred_devices.append("cpu")

        last_err = None
        
        for dev in preferred_devices:
            try:
                self.logger.info(f"Loading '{model_name}' model on {dev}.")
                self._device = dev
                self._processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
                self._model = AutoModelForVision2Seq.from_pretrained(model_name).to(self._device)
                self._model.eval()
                self.logger.info(f"Successfully loaded on {dev}.")
                break
            except RuntimeError as e:
                last_err = e
                self.logger.error(f"Failed to load on {dev}: {e}")
            except Exception as e: 
                last_err = e
                self.logger.error(f"An unexpected error occurred while loading model on {dev}: {e}")
        
        if self._model is None: #check if nothing works, then raise error
            raise RuntimeError(f"Could not load model on any device. Last error:\n{last_err}")
            
    @property
    def device(self):
        return self._device

    @property
    def processor(self):
        return self._processor

    @property
    def model(self):
        return self._model

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

        inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_length=max_length, num_beams=num_beams, early_stopping=early_stopping)

        # text return
        return self._processor.decode(outputs[0], skip_special_tokens=True)

    def __enforce_type(self, value, expected_type, arg_name):
        if not isinstance(value, expected_type):
            raise TypeError(f"Argument '{arg_name}' must be of type {expected_type.__name__}, but received {type(value).__name__}")

#Example
if __name__ == "__main__":
    _cap = ImageCaptioner(device="cuda")
    _text = _cap.write_caption("img.jpg")
    print("Caption:", _text)

