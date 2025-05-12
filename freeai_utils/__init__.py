from .audio_record import WavRecorder, MP3Recorder
from .audio_to_text_whisper import OpenAIWhisper
from .clean_text_for_tts import clean_ai_text_for_tts
from .geminiAPI import GeminiClient
from .google_search import GoogleSearcher
from .image_to_text import ImageCaptioner
from .text_from_image_easyocr import Text_Extractor_EasyOCR
from .text_to_speech_gtts import gtts_print_supported_languages, gtts_speak
from .text_to_speech_pyttsx3 import Text_To_Speech_Pyttsx3

__all__ = [
    "WavRecorder", "MP3Recorder",
    "OpenAIWhisper",
    "clean_ai_text_for_tts",
    "GeminiClient",
    "GoogleSearcher",
    "ImageCaptioner",
    "Text_Extractor_EasyOCR",
    "gtts_print_supported_languages", "gtts_speak",
    "Text_To_Speech_Pyttsx3",
    "PDF_DOCX_Reader",
]