import torch
from diffusers import AutoPipelineForText2Image

class TurboImage:
    def __init__(self, model_name : str = "stabilityai/sdxl-turbo", device : str = None):
        
        self.model = AutoPipelineForText2Image.from_pretrained(model_name,
                                                        torch_dtype=torch.float32 if device == "cpu" else torch.float16, # Use float32 for CPU
                                                        device_map=None)
    
    def create_images(self, something):
        pass