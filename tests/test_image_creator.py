import pytest
import gc
from freeai_utils.image_creator import SDXL_TurboImage, SD15_Image

def test_init_sdxl(): #idk what to test...
    model = SDXL_TurboImage()
    assert model.model is not None
    assert model.device in ("cpu", "cuda")
    model.generate_images("A house")
    del model
    gc.collect()

@pytest.fixture(scope="module")
def sd15():
    model = SD15_Image(embed_default=False)
    yield model
    del model
    gc.collect()

@pytest.fixture(autouse=True)#run after each functions
def test_create_img(request):
    # List of test names dont want this test to run after
    trigger_after = {"test_init_sdxl", "sd15", "test_init_sd15", "test_load_default_embed_lora_sd15", "test_clean_up"}

    if request.function.__name__ not in trigger_after:
        print(f"Creating image before test: {request.function.__name__}")
        sd15 = request.getfixturevalue("sd15")
        prompt = "A house"
        sd15.generate_images(prompt)
        sd15.generate_images(prompt, extra_detail = 1)

def test_init_sd15(sd15):
    assert sd15.model is not None
    assert sd15.device in ("cpu", "cuda")
    assert sd15.model.safety_checker is not None
    assert sd15.model.feature_extractor is not None
    #because embed default is false
    embeds = ["easynegative", "badprompt", "negativehand"]
    for embed in embeds:
        assert embed not in sd15.model.tokenizer.added_tokens_encoder
    assert sd15._model.get_active_adapters() == []

def test_load_default_embed_lora_sd15(sd15):
    sd15._load_default_embed_and_lora()
    embeds = ["easynegative", "badprompt", "negativehand"]
    for embed in embeds:
        assert embed in sd15.model.tokenizer.added_tokens_encoder
    assert sd15._model.get_active_adapters() == ["add_detail"]


def test_scheduler_sd15_sdekarras(sd15):
    sd15._custom_scheduler("SDE Karras")
    from diffusers import DPMSolverSinglestepScheduler
    # build the DPM++ SDE Karras scheduler
    new_scheduler  = DPMSolverSinglestepScheduler.from_config( #this requires number of step to be even
        sd15._model.scheduler.config,
        use_karras_sigmas=True,
        lower_order_final=True
    )
    assert sd15.model.scheduler.config == new_scheduler.config

def test_scheduler_sd15_dpmkarras(sd15):
    sd15._custom_scheduler("DPM++ 2M Karras")
    from diffusers import DPMSolverMultistepScheduler
    new_scheduler  = DPMSolverMultistepScheduler.from_config( 
            sd15._model.scheduler.config,
            use_karras_sigmas=True,
            solver_order=2,
            algorithm_type="dpmsolver++",
            lower_order_final=True 
        )
    assert sd15.model.scheduler.config == new_scheduler.config
    
def test_scheduler_sd15_euler_a(sd15):
    sd15._custom_scheduler("Euler A")
    from diffusers import EulerAncestralDiscreteScheduler
    new_scheduler  = EulerAncestralDiscreteScheduler.from_config(sd15._model.scheduler.config, use_karras_sigmas=True,
        lower_order_final = True)
    assert sd15.model.scheduler.config == new_scheduler.config

def test_enable_safety_sd15(sd15):
    sd15.enable_safety()
    from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker #safety check
    from transformers import CLIPImageProcessor #for safety check
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-safety-checker"
    ).to(sd15._device)
    feature_extractor = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )
    assert sd15._model.safety_checker.config.to_dict() == safety_checker.config.to_dict()
    assert sd15._model.feature_extractor.to_dict() == feature_extractor.to_dict()

def test_disable_safety_sd15(sd15):
    sd15.disable_safety()
    assert sd15.model.safety_checker is None
    assert sd15.model.feature_extractor is None
    
def test_clean_up():
    from freeai_utils import Cleaner
    import os
    cur = os.path.abspath(os.path.join(os.getcwd(), "generated_images"))
    cleaner = Cleaner(directory=cur)
    cleaner.remove_all_files_end_with(".png")