import whisper #need pip install openai-whisper
import soundfile as sf #need pip install souldfile
import numpy as np # need pip install numpy
import librosa #need pip install librosa 
import torch # need pip install torch
from typing import Dict, Any
from typing import Optional
from freeai_utils.log_set_up import setup_logging

# 3 function use: transcribe -> return Dict (return everything and you choose which to get)
#                 get_lang_detect -> return str (return the language)
#                 get_translation -> return str (return the translation only, for people who doesnt care what language or anything else)

class OpenAIWhisper:
    
    __slots__ = ("_model", "_initialized", "logger", "_device", "_sample_rate")
    
    def __init__(self, model: str = "medium", sample_rate: int = 16000, device : Optional[str]  = None) -> None:
        #check type
        self.__enforce_type(sample_rate, int, sample_rate)
        self.logger = setup_logging(self.__class__.__name__)
        
        # init not lock
        super().__setattr__("_initialized", False)
        
        # model: Whisper model size ("tiny", "base", "small", "medium", "large")
        self._sample_rate = sample_rate
        self._model = None 
        self._device = None 
        
        #init the var to hold device available
        preferred_devices = []
        
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

        last_err = None
        for dev in preferred_devices:
            try:
                self.logger.info(f"Loading '{model}' model on {dev}...")
                self._model = whisper.load_model(model, device=dev)
                self._device = dev
                self.logger.info(f"Model successfully loaded on {dev}.")
                break
            except RuntimeError as e:
                last_err = e
                self.logger.error(f"Failed to load on {dev}: {e}")
            except Exception as e: 
                last_err = e
                self.logger.error(f"An unexpected error occurred while loading model on {dev}: {e}")
        
        if self._model is None: #check if nothing works, then raise error
            raise RuntimeError(f"Could not load model on any device. Last error:\n{last_err}")
        
        # lock
        super().__setattr__("_initialized", True)
        
    @property
    def sample_rate(self):
        # Returns the sample rate used by the model.
        return self._sample_rate

    @property
    def model(self):
        # Returns the loaded Whisper model.
        return self._model

    @property
    def device(self):
        # Returns the device the model is loaded on.
        return self._device
    
    def _load_wav(self, path: str) -> np.ndarray:
        try:
            audio, orig_sr = sf.read(path)
        except FileNotFoundError:
            self.logger.error(f"Error: Audio file not found at {path}")
            raise
        except Exception as e:
            self.logger.error(f"Error reading audio file {path}: {e}")
            raise

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1) # Change to mono
        
        if orig_sr != self._sample_rate:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self._sample_rate)
        
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
            audio_data = self._load_wav(audio_path)
                
            # get the np.ndarray to the transcribe
            self.logger.info(f"Transcribing audio from {audio_path} on {self._device}...")
            result = self._model.transcribe(audio_data, fp16=fp16 if self._device == "cuda" else False, **transcribe_kwargs)
            self.logger.info(f"Transcription successful.")
            return result #return result
        except Exception as e:
            self.logger.error(f"Transcription failed for {audio_path}: {e}")
            raise
    
    def get_translation(self, audio_path: str, fp16: bool = False, **transcribe_kwargs: Any) -> str:
        text = self.transcribe(audio_path=audio_path, fp16=fp16, **transcribe_kwargs)["text"]
        return text

    def get_lang_detect(self, audio_path: str, fp16: bool = False, **transcribe_kwargs: Any) -> str:
        language = self.transcribe(audio_path=audio_path, fp16=fp16, **transcribe_kwargs)["language"]
        return language
    
    def __enforce_type(self, value, expected_type, arg_name):
        if not isinstance(value, expected_type):
            raise TypeError(f"Argument '{arg_name}' must be of type {expected_type.__name__}, but received {type(value).__name__}")
    
    def __setattr__(self, name, value):
        # once initialized, block these core attributes
        if getattr(self, "_initialized", False) and name in ("_model", "_device"):
            raise AttributeError(f"Cannot reassign '{name}' after initialization")
        super().__setattr__(name, value)