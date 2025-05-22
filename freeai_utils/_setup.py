def install_default_model():
    decision = input("""This function will download all default models used by this library. 
The process can take a significant amount of time, so feel free to take a break.
Would you like to proceed with the download? (Y/n): """)
    if decision.lower() == "y":
        from .audio_to_text_vn import VN_Whisper
        from .audio_to_text_whisper import OpenAIWhisper
        from .decider import DecisionMaker
        from .document_filter import DocumentFilter
        from .image_to_text import ImageCaptioner
        from .text_from_image_easyocr import Text_Extractor_EasyOCR
        from .language_detection import MBartTranslator, M2M100Translator
        vnwhisper = VN_Whisper()
        whisper = OpenAIWhisper()
        dec = DecisionMaker()
        doc = DocumentFilter(auto_init=False)
        imgdes = ImageCaptioner()
        textocr = Text_Extractor_EasyOCR()
        mbart = MBartTranslator()
        m2m100 = M2M100Translator()
    else:
        print("Download cancelled.")
       
if __name__ == "__main__":
    install_default_model()