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
        vnwhisper = VN_Whisper()
        whisper = OpenAIWhisper()
        dec = DecisionMaker()
        doc = DocumentFilter()
        imgdes = ImageCaptioner()
        textocr = Text_Extractor_EasyOCR()
    else:
        print("Download cancelled.")
       
if __name__ == "__main__":
    install_default_model()