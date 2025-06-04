import torch
from diffusers import AutoPipelineForText2Image #for sdxl_turbo
import os
from diffusers import StableDiffusionPipeline #for sd1.5
from transformers import CLIPTokenizer #for sd1.5
from freeai_utils.log_set_up import setup_logging
import logging
from typing import Union, Optional

class SDXL_TurboImage:
    
    __slots__ = ("_model", "_device", "logger", "_initialized")
    
    _model: AutoPipelineForText2Image
    _device: str
    logger: logging.Logger
    _initialized: bool
    
    def __init__(self, model_name : str = "stabilityai/sdxl-turbo", device : str = None):
        #check type
        self.__enforce_type(model_name, str, "model_name")
        
        #init
        super().__setattr__("_initialized", False)
        
        self.logger = setup_logging(self.__class__.__name__)
        #init the var to hold device available
        preferred_devices = []
        self._model = None
        #try input first
        if device is not None:
            self.__enforce_type(device, str, "device")
            preferred_devices.append(device)
        
        # try cuda second 
        if torch.cuda.is_available() and "cuda" not in preferred_devices:
            preferred_devices.append("cuda")
        
        # fall back to CPU if not already there
        if "cpu" not in preferred_devices:
            preferred_devices.append("cpu")
        
        for dev in preferred_devices:
            dtype = torch.float32 if dev == "cpu" else torch.float16
            try:
                self.logger.info(f"Loading '{model_name}' on {dev}")
                self._model = AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=dtype,).to(dev)
                self._device = dev
                self.logger.info(f"Successfully loaded on {dev}")
                break
            except Exception as e:
                self.logger.error(f"Failed to load on {dev}: {e}")
                
        if self._model is None:
            raise RuntimeError(f"Cannot load '{model_name}' on any of: {preferred_devices}")
        
        #lock
        super().__setattr__("_initialized", True)
        
        self.logger.info("Fast image generator, sacrifice quality for speed")

    @property
    def device(self):
        return self._device
    
    @property
    def model(self):
        return self._model

    def generate_images(self, prompt : str = None, 
                    negative_prompt : str = """lowres, text, error, missing fingers, extra digit, missing libs, cropped, bad mouth, bad lips
                                                worst quality, lowres, (bad anatomy), (bad hands), text, missing finger, 
                                                extra digits, 2 eyes color, blurry, bad eyes, low quality, 3d, sepia, painting, 
                                                cartoons, sketch, signature, watermark, username""", 
                    steps : int = 2,
                    number_of_images : int = 2,
                    image_name : str = "generated_image",
                    output_dir : str = "generated_images") -> None:
        
        #check type
        self.__enforce_type(prompt, str, "prompt")
        self.__enforce_type(negative_prompt, str, "negative_prompt")
        self.__enforce_type(steps, int, "steps")
        self.__enforce_type(number_of_images, int, "number_of_images")
        self.__enforce_type(image_name, str, "image_name")
        self.__enforce_type(output_dir, str, "output_dir")
        
        #make the folder exists
        os.makedirs(output_dir, exist_ok=True)
        try:
            output = self._model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=0.0,
                num_images_per_prompt=number_of_images
            )
        except RuntimeError:
            raise RuntimeError(f"""Look like your computer can't handle the image generation. Please lower your 'steps' and 'number_of_images'.
                                 Your current input for steps: {steps}
                                 Your current input for images per generation: {number_of_images}
                                 """)
        except Exception as e:
            self.logger.critical("Unknown error")
            raise
        
        images = output.images  # a list of PIL images
        
        for i, img in enumerate(images):
            img.save(f"{output_dir}\\{image_name}{i}.png") #will add time later to not be overload
        print(f"Saved {len(images)} images.")

    def __setattr__(self, name, value):
        # once initialized, block these core attributes
        if getattr(self, "_initialized", False) and name in ("_model", "_device"):
            raise AttributeError(f"Cannot reassign '{name}' after initialization")
        super().__setattr__(name, value)
        
    def __enforce_type(self, value, expected_type, arg_name):
        if not isinstance(value, expected_type):
            raise TypeError(f"Argument '{arg_name}' must be of type {expected_type.__name__}, but received {type(value).__name__}")

class SD15_Image:
    
    logger : logging.Logger
    _model : StableDiffusionPipeline
    _tokenizer : CLIPTokenizer
    
    def __init__(self, preferred_device : Optional[str] = None ,support_model : str = "" , model_path : str = None, scheduler : str = "default"):
        #model path is to check whether get from lib or get from running folder
        #support_model is for use or not
        #schedule is for sample
        self.logger = setup_logging(self.__class__.__name__)
        self.logger.info("Init...")
        self._device = None
        self._model = None
        
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloaded_models")
        else:
            self.__enforce_type(model_path, str, "model_path")
            
        path = os.path.join(model_path, support_model) #init the path to model
        
        preferred_devices = []

        #try input first
        if preferred_device is not None:
            self.__enforce_type(preferred_device, str, "device")
            preferred_devices.append(preferred_device)
        # try cuda second 
        if torch.cuda.is_available() and "cuda" not in preferred_devices:
            preferred_devices.append("cuda")
        # fall back to CPU if not already there
        if "cpu" not in preferred_devices:
            preferred_devices.append("cpu")
        
        if support_model.strip() == "":
            self.logger.info("No support_model specify, using default stable-diffusion-v1-5, skipping all other configures..")
            self._default_setup(preferred_devices)
        else:
            self.logger.info("Custom setup for SD1.5")
            self._custom_setup(preferred_devices = preferred_devices, path = path, scheduler = scheduler)
        #embeded support
        
        if self._model is None:
            raise RuntimeError(f"Could not load model on any device: {preferred_devices}")
        else:
            self.logger.info(f"Successfully loaded on {self._device}")
        
        embedding_paths = {
            "easynegative": os.path.join(model_path, "easynegative.safetensors"),
            "badprompt": os.path.join(model_path, "bad_prompt.pt"),
            "negativehand": os.path.join(model_path, "negative_hand.pt")
        }
        
        for token_name, epath in embedding_paths.items(): #load embeded path to model
            try:
                self._model.load_textual_inversion(epath, token=token_name)
            except Exception:
                self.logger.error(f"Fail to load {token_name} at {epath}.\n May be you wanna try 'freeai-utils setup ICE' to download the file?")
                raise
            
        self.logger.info("Successfully Initialized")
    
    def generate_images(self,
                        positive_prompt : str = None, 
                        negative_prompt : str = "<easynegative:0.8>, <negativehand:2.1>, <badprompt:1.4>",
                        image_name : str = "generated_image", 
                        output_dir : str = "generated_images",
                        width : int = 512, 
                        height : int = 512, 
                        steps : int = 30, 
                        guidance_scale : float = 8, 
                        number_of_images : int = 2, 
                        seed : int = 123456789) -> None:
        
        generator = torch.Generator(self._device).manual_seed(seed)
        os.makedirs(output_dir, exist_ok=True) #make the dir exists
        try:
            with torch.inference_mode():
                output = self._model(
                    prompt= positive_prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps= steps, 
                    guidance_scale= guidance_scale,
                    num_images_per_prompt= number_of_images,
                    generator=generator,
                )
        except RuntimeError:
            raise RuntimeError(f"""Look like your computer can't handle the image generation. Please lower your 'steps' or 'guidance_scale' or 'number_of_images'.
                                 Your current input for steps: {steps}
                                 Your current input for guidance_scale : {guidance_scale}
                                 Your current input for images per generation: {number_of_images}
                                 """)
        except Exception as e:
            self.logger.critical("Unknown error")
            raise
            
        for i, img in enumerate(output.images):
            img.save(f"{output_dir}\\{image_name}{i}.png") #will add time later to not be overload
            print(f"Saved {len(output.images)} images.")
    
    def _help_config(self) -> None:
        model_list = ["anime_pastal_dream.safetensors", "meinapastel.safetensors", "reality.safetensors", "annylora_checkpoint.safetensors"]
        scheduler_list = ["default", "SDE Karras"]
        print("*" * 40)
        print("Including support_model: ")
        for item in model_list:
            print(item)
        print("*" * 40)
        print("Including scheduler: ")
        for item in scheduler_list:
            print(item)
        print("*" * 40)
        
    def _default_setup(self, preferred_devices):
        for dev in preferred_devices:
            try:
                self.logger.info(f"Loading on {dev}")
                self._model = StableDiffusionPipeline.from_pretrained(
                    "stable-diffusion-v1-5/stable-diffusion-v1-5",
                    torch_dtype=torch.float32 if dev == "cpu" else torch.float16,
                ).to(dev)
                self._device = dev
                self._model.enable_attention_slicing() #reduce size
                break
            except Exception as e:
                self.logger.error(f"Failed to load on {dev}: {e}")

    def _custom_setup(self, preferred_devices, path, scheduler):
        self.logger.info(f"Loading support model at {path}")
        self.logger.info(f"Loading scheduler: {scheduler if scheduler != "default" else "Euler"}")
        # Load tokenizer
        self._tokenizer = CLIPTokenizer.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            subfolder="tokenizer"
        )
        #iterate through to see which allow
        for dev in preferred_devices:
            try:
                self.logger.info(f"Loading on {dev}")
                self._model = StableDiffusionPipeline.from_single_file( #init model
                    path,
                    tokenizer=self._tokenizer,
                    torch_dtype=torch.float32 if dev == "cpu" else torch.float16,
                    safety_checker=None,
                    mean_resizing = False,
                ).to(dev)
                self._device = dev
                self._model.enable_attention_slicing() #reduce size
                break
            except Exception as e:
                self.logger.error(f"Failed to load on {dev}: {e}")
        
        if self._model is None:
            raise RuntimeError(f"Could not load model on any device: {preferred_devices}") #raise before config scheduler
        
        #schedule type
        if scheduler == "SDE Karras":
            from diffusers import DPMSolverSinglestepScheduler
            # build the DPM++ SDE Karras scheduler
            sde_karras = DPMSolverSinglestepScheduler.from_config( #this requires number of step to be even
                self._model.scheduler.config,
                use_karras_sigmas=True,
                lower_order_final = True
            )
            self._model.scheduler = sde_karras
         
        else: #default euler
            from diffusers import EulerDiscreteScheduler
            self._model.scheduler = EulerDiscreteScheduler.from_config(self._model.scheduler.config)
    
    def __enforce_type(self, value, expected_type, arg_name):
        if not isinstance(value, expected_type):
            raise TypeError(f"Argument '{arg_name}' must be of type {expected_type.__name__}, but received {type(value).__name__}")

class SDXL10_Image:
    def __init__(self):
        print("nothing here yet :D")
        pass

if __name__ == "__main__":
    import gc
    # imagegenerator = SDXL_TurboImage(device="cuda")
    img2 = SD15_Image()
    img2.generate_images("Create an image of an blue hair anime girl", number_of_images=1)
    img2 = None
    gc.collect()
    img3 = SD15_Image(support_model="annylora_checkpoint.safetensors", scheduler="SDE Karras")
    # img2._help_config()
    img3.generate_images("Create an image of an blue hair anime girl", number_of_images=2)
    # imagegenerator.generate_images(prompt= "Create an image of an blue hair anime girl", image_name="generated_image", output_dir="images")