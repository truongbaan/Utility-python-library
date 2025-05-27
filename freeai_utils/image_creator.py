import torch
from diffusers import AutoPipelineForText2Image
import os

class SDXL_TurboImage:
    def __init__(self, model_name : str = "stabilityai/sdxl-turbo", device : str = None):
        
        self.device = device if device else "cpu"
        
        self.model = AutoPipelineForText2Image.from_pretrained(model_name,
                                                        torch_dtype=torch.float32 if self.device == "cpu" else torch.float16, # Use float32 for CPU
                                                        device_map=None)
        self.model.to(self.device)
        print("Fast image generator, sacrifice for quality")
    
    def generate_images(self, prompt : str = None, 
                    negative_prompt : str = """lowres, text, error, missing fingers, extra digit, missing libs, cropped, bad mouth, bad lips
                                                worst quality, lowres, (bad anatomy), (bad hands), text, missing finger, 
                                                extra digits, 2 eyes color, blurry, bad eyes, low quality, 3d, sepia, painting, 
                                                cartoons, sketch, signature, watermark, username""", 
                    steps : int = 2,
                    number_of_images : int = 2,
                    image_name : str = "generated_image",
                    output_dir : str = "") -> None:
        output = self.model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=0.0,
            num_images_per_prompt=number_of_images
        )
        images = output.images  # a list of PIL images
        output_dir = output_dir.replace("\"", "").replace("/", "") #remove before to ensure it is correct
        os.makedirs(output_dir, exist_ok=True)
        
        for i, img in enumerate(images):
            img.save(f"{output_dir}\\{image_name}{i}.png") #will add time later to not be overload
        print(f"Saved {len(images)} images.")
        
        
class SD15_Image:
    def __init__(self, preferred_device ,support_model ,output_dir, model_path, scheduler):
        #model path is to check whether get from lib or get from running folder
        #support_model is for use or not
        #schedule is for sample
        pass
    
    def generate_images(self,positive_prompt, negative_prompt,filename, width, height, steps, guidance_scale, number_of_images, seed) -> None:
        pass
    
if __name__ == "__main__":
    imagegenerator = SDXL_TurboImage(device="cpu")
    imagegenerator.generate_images(prompt= "Create an image of an blue hair anime girl", image_name="generated_image", output_dir="images")