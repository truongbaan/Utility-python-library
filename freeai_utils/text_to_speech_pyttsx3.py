import pyttsx3 #need pip install pyttsx3

class Text_To_Speech_Pyttsx3:
    def __init__(self, rate=170, volume=1.0, voice_index=0) -> None:
        self._engine = pyttsx3.init()
        self.config_voice(rate, volume, voice_index)

    def config_voice(self, rate: float = 170, volume: float = 1.0, voice_index: int = 0) -> None:
        if not isinstance(rate, (int, float)):
            raise TypeError("Rate must be a number.")
        if not (0.0 <= volume <= 1.0):
            raise ValueError("Volume must be between 0.0 and 1.0.")
        
        voices = self._engine.getProperty('voices')
        if not isinstance(voice_index, int) or not (0 <= voice_index < len(voices)):
            raise ValueError(f"Voice index must be between 0 and {len(voices)-1}")
        
        self._engine.setProperty('rate', rate)
        self._engine.setProperty('volume', volume)
        self._engine.setProperty('voice', voices[voice_index].id)

    def speak(self, text: str) -> None:
        if not isinstance(text, str):
            raise ValueError(f"The provided text is not string!")
        self._engine.say((text))
        self._engine.runAndWait()

    def stop(self) -> None:
        self._engine.stop()