import whisper #need pip install openai-whisper
import soundfile as sf #need pip install souldfile
import numpy as np # need pip install numpy
import librosa #need pip install librosa 
import torch # need pip install torch
from typing import Dict, Any

# 3 function use: transcribe -> return Dict (return everything and you choose which to get)
#                 get_lang_detect -> return str (return the language)
#                 get_translation -> return str (return the translation only, for people who doesnt care what language or anything else)
class OpenAIWhisper:
    def __init__(self, model_size: str = "medium", sample_rate: int = 16000) -> None:
        # model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
        self.sample_rate = sample_rate
        self.model = None 
        self.device = None 
        
        #check type
        self.__enforce_type(sample_rate, int, sample_rate)
        
        # check GPU first, fall back to CPU if not work
        preferred_devices = []
        if torch.cuda.is_available():
            preferred_devices.append("cuda")
        preferred_devices.append("cpu")

        last_err = None
        for dev in preferred_devices:
            try:
                print(f"[OpenAIWhisper] Loading '{model_size}' model on {dev}...")
                self.model = whisper.load_model(model_size, device=dev)
                self.device = dev
                print(f"[OpenAIWhisper] Model successfully loaded on {dev}.")
                break
            except RuntimeError as e:
                last_err = e
                print(f"[OpenAIWhisper] Failed to load on {dev}: {e}")
            except Exception as e: 
                last_err = e
                print(f"[OpenAIWhisper] An unexpected error occurred while loading model on {dev}: {e}")
        
        if self.model is None: #check if nothing works, then raise error
            raise RuntimeError(f"Could not load model on any device. Last error:\n{last_err}")

    def _load_wav(self, path: str) -> np.ndarray:
        try:
            audio, orig_sr = sf.read(path)
        except FileNotFoundError:
            print(f"[OpenAIWhisper] Error: Audio file not found at {path}")
            raise
        except Exception as e:
            print(f"[OpenAIWhisper] Error reading audio file {path}: {e}")
            raise

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1) # Chuyển sang mono
        
        if orig_sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sample_rate)
        
        return audio.astype(np.float32)

    def transcribe(self, audio_path: str, fp16: bool = False, **transcribe_kwargs: Any) -> Dict[str, Any]:
        # Transcribes audio from a file path.
        # param audio_path: Path to the audio file.
        # param fp16: Whether to use fp16 precision (GPU only).
        # param transcribe_kwargs: Additional arguments dokter for whisper.model.transcribe().
        # return: A dictionary containing the transcription results.
        
        self.__enforce_type(audio_path, str, "audio_path")
        self.__enforce_type(fp16, bool, "fp16")
        
        try:
            # load file
            file_dir = os.path.dirname(os.path.abspath(__file__)) #get correct place
            audio_path = os.path.join(file_dir, audio_path)
            audio_data = self._load_wav(audio_path)
            
            # get the np.ndarray to the transcribe
            print(f"[OpenAIWhisper] Transcribing audio from {audio_path} on {self.device}...")
            result = self.model.transcribe(audio_data, fp16=fp16 if self.device == "cuda" else False, **transcribe_kwargs)
            print(f"[OpenAIWhisper] Transcription successful.")
            return result #return result
        except Exception as e:
            print(f"[OpenAIWhisper] Transcription failed for {audio_path}: {e}")
            raise
    
    def get_translation(self, audio_path: str, fp16: bool = False, **transcribe_kwargs: Any) -> str:
        text = self.transcribe(audio_path=audio_path, fp16=fp16, **transcribe_kwargs)["text"]
        return text

    def get_lang_detect(self, audio_path: str, fp16: bool = False, **transcribe_kwargs: Any):
        language = self.transcribe(audio_path=audio_path, fp16=fp16, **transcribe_kwargs)["language"]
        return language
    
    def __enforce_type(self, value, expected_type, arg_name):
        if not isinstance(value, expected_type):
            raise TypeError(f"Argument '{arg_name}' must be of type {expected_type.__name__}, but received {type(value).__name__}")
        
# Example usage      
if __name__ == "__main__":
    import os
    FILENAME = "sample\\output_pyaudio.wav"

    _transcriber = OpenAIWhisper()
    _output = _transcriber.transcribe(FILENAME)
    
    #usually, we only care about 2 thing, the transcribe and the language detect
    print("Detected language:", _output["language"])
    print("Transcript:", _output["text"])
    
    #everything in the output for anyone cares
    print(_output)
    
    _language_dect =_transcriber.get_lang_detect(FILENAME)
    print("Language used: ")
    print(_language_dect)
    
    _text_translated = _transcriber.get_translation(FILENAME)
    print("Text transcription only: ")
    print(_text_translated)
