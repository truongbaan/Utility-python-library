import time
import json
import numpy as np
import sounddevice as sd #need pip install sounddevice
from vosk import Model, KaldiRecognizer #need pip install vosk
from typing import Optional
import os

#this model is for live transcription than using whisper model to transcribe a provided audio (speech to text)
class STT_Vosk:
    def __init__(self, model_name  : str = 'en_us_015', model_path : Optional[str] = None, sample_rate : int = 16000, dtype : str = "int16", channels  : int  = 1, frame_duration : float = 0.1) -> None:
        #check type before init
        self.__enforce_type(model_name, str, "model_name")
        self.__enforce_type(model_path, (str, type(None)), "model_path")
        self.__enforce_type(sample_rate, int, "sample_rate")
        self.__enforce_type(dtype, str, "dtype")
        self.__enforce_type(channels, int, "channels")
        self.__enforce_type(frame_duration, (int, float), "frame_duration")
        
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vosk_models")#use defualt downloaded
        model = os.path.join(model_path, model_name) #place where the folder is
        if not os.path.isdir(model):
            raise FileNotFoundError(f"Error, path not found or is not a directory. Path : {model}")
        
        self._model = Model(model)
        self._rec = KaldiRecognizer(self._model, sample_rate)
        self._sample_rate = sample_rate
        self._dtype = dtype
        self._channels = channels
        self._frame_duration = frame_duration
        self._blocksize = int(sample_rate * self._frame_duration)
        pass
    
    def live_transcribe_toggle(toggle_off : str = "`"):
        pass
    
    def live_transcribe_until_silence(self, silence_thresh: float = 0.01, silence_duration: float = 4.0) -> str:
        stream = sd.RawInputStream(
            samplerate=self._sample_rate,
            blocksize=self._blocksize,
            dtype=self._dtype,
            channels= self._channels
        )
        stream.start()
        print(f"🎙 Listening… (will stop after {silence_duration}s of silence)")

        last_voice = time.time()
        last_partial = ""
        segments = []

        try:
            while True:
                data, overflowed = stream.read(self._blocksize)
                if overflowed:
                    # Just drop it and keep going
                    continue

                raw = bytes(data)
                self._rec.AcceptWaveform(raw)  # Always call, but ignore its True/False return

                # 1) Live partial
                p = json.loads(self._rec.PartialResult()).get("partial", "")
                if p != last_partial:
                    print(f"🗣 {p}", end="\r")
                    last_partial = p

                # 2) Silence detection
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                rms = np.sqrt((audio**2).mean()) / np.iinfo(np.int16).max
                # print(rms)
                if rms > silence_thresh:
                    last_voice = time.time()

                if time.time() - last_voice > silence_duration:
                    print("\n🔇 Silence reached, finalizing…")
                    break

        except KeyboardInterrupt:
            print("\n⛔ Interrupted by user.")
        finally:
            stream.stop()
            stream.close()

        # Now flush out the final text
        final_txt = json.loads(self._rec.FinalResult()).get("text", "")
        if final_txt:
            print(f"✅ Final: {final_txt}")
            segments.append(final_txt)

        return " ".join(segments).strip()
    
    def rms(self, data): #use for silence
        return np.sqrt(np.mean(np.square(data / np.iinfo(np.dtype(self._dtype)).max)))

    def __enforce_type(self, value, expected_types, arg_name):
        if not isinstance(value, expected_types):
            expected_names = [t.__name__ for t in expected_types] if isinstance(expected_types, tuple) else [expected_types.__name__]
            expected_str = ", ".join(expected_names)
            raise TypeError(f"Argument '{arg_name}' must be of type {expected_str}, but received {type(value).__name__}")
        
if __name__ == "__main__":
    test = STT_Vosk()
    talk = test.live_transcribe_until_silence(silence_duration=4)
    print(talk)