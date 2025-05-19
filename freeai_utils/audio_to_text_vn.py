import torch
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers import logging as transformers_logging
from typing import Union
from freeai_utils.log_set_up import setup_logging

#other model_id: "namphungdn134/whisper-base-vi"
class VN_Whisper:
    def __init__(self, model_id="namphungdn134/whisper-small-vi", device : Union[str, None] = None) -> None:
        self._model = None 
        self._device = None 
        self.logger = setup_logging(self.__class__.__name__)
        
        #init the var to hold device available
        preferred_devices = []
        
        #try input first
        if device is not None:
            preferred_devices.append(device)
        
        # try cuda second 
        if torch.cuda.is_available() and "cuda" not in preferred_devices:
            preferred_devices.append("cuda")

        # fall back to CPU if not already there
        if "cpu" not in preferred_devices:
            preferred_devices.append("cpu")
            
        for dev in preferred_devices:
            try:
                self.logger.info(f"Loading '{model_id}' model on {dev}...")
                # Load processor and model
                self.processor = AutoProcessor.from_pretrained(model_id)
                self._model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(dev)
                self.logger.info(f"Model successfully loaded on {dev}.")
                self._device = dev
                break
            except RuntimeError as e:
                self.logger.error(f"Failed to load on {dev}: {e}")
            except Exception as e: 
                self.logger.error(f"An unexpected error occurred while loading model on {dev}: {e}")
        
        if self._model is None: #check if nothing works, then raise error
            raise RuntimeError(f"Could not load model on any device")
        # Suppress transformer-related warnings
        transformers_logging.set_verbosity_error()
    
    @property
    def model(self):
        return self._model  
    
    @property
    def device(self):
        return self._device
              
    def transcribe_audio(self, audio_path) -> str:
        # Load and preprocess audio
        self.logger.info(f"Loading audio from: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=16000)
        input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(self._device)

        # Generate transcription
        self.logger.info("Generating transcription...")
        with torch.no_grad():
            predicted_ids = self._model.generate(input_features, num_beams=1, use_cache=True)

        # Decode transcription
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        transcription = self.__verify(transcription)
        return transcription

    def __verify(self, prompt : str) -> str:
        # 1.this to check if nothing is heard
        # 2. that phrase looks like the default model use it as temp phrase when nothing is heard, probably to ensure the audio when heard nothing would return a str
        # 3. this check to make sure it doesnt just get some 1 word speech and translate to that
        if prompt.strip() == "" or prompt.strip() == "thôi để con đo huyết áp trước cái" or len(prompt) < 10: 
            return ""
        else: return prompt
