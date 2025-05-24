import gc

def install_model(target):
    if target is None or target == "A":
        install_default_model()  # Installs all default models (excluding image generation)
    elif target == "S":
        print("Downloading default models for speech-to-text")
        download_transcription()
    elif target == "D":
        print("Downloading default models for document processing")
        download_document_related()
    elif target == "I":
        print("Downloading default models for OCR and image captioning")
        download_image_related()
    elif target == "T":
        print("Downloading default models for translation")
        download_translation()
    elif target == "L":
        print("Downloading default models for local LLMs")
        download_LLM()
    else:
        print("No cmd found, please try again")
        
def install_default_model():
    decision = input(
        "This function will download all default models used by this library.\n"
        "The process can take a significant amount of time, so feel free to take a break.\n"
        "Would you like to proceed with the download? (Y/n): ").strip().lower()
        
    if decision != "y":
        print("Download cancelled.")
        return
        
    download_transcription()
    download_document_related()
    download_image_related()
    download_translation()

def download_transcription():
    from .audio_to_text_vn import VN_Whisper
    from .audio_to_text_whisper import OpenAIWhisper
     
    _download_and_purge(VN_Whisper)
    _download_and_purge(OpenAIWhisper)


def download_document_related():
    from .decider import DecisionMaker
    from .document_filter import DocumentFilter
     
    _download_and_purge(DecisionMaker)
    _download_and_purge(DocumentFilter, auto_init=False)


def download_image_related():
    from .image_to_text import ImageCaptioner
    from .text_from_image_easyocr import Text_Extractor_EasyOCR
    
    _download_and_purge(ImageCaptioner)
    _download_and_purge(Text_Extractor_EasyOCR)


def download_translation():
    from .language_detection import MBartTranslator, M2M100Translator
    
    _download_and_purge(MBartTranslator)
    _download_and_purge(M2M100Translator)

def _download_and_purge(cls, *args, **kwargs):
    inst = cls(*args, **kwargs)
    del inst
    gc.collect()

def download_LLM():
    pass #not yet

def download_image_creation_related():
    from .image_creator import TurboImage
    _download_and_purge(TurboImage)

def _download_in_civitai():
    pass

if __name__ == "__main__":
    install_default_model()