import pyttsx3 #need pip install pyttsx3
import traceback

class Text_To_Speech_Pyttsx3:
    def __init__(self, rate=170, volume=1.0, voice_index=0) -> None:
        self.engine = pyttsx3.init()
        self.config_voice(rate, volume, voice_index)

    def config_voice(self, rate: float = 170, volume: float = 1.0, voice_index: int = 0) -> None:
        if not isinstance(rate, (int, float)):
            raise TypeError("Rate must be a number.")
        if not (0.0 <= volume <= 1.0):
            raise ValueError("Volume must be between 0.0 and 1.0.")
        
        voices = self.engine.getProperty('voices')
        if not isinstance(voice_index, int) or not (0 <= voice_index < len(voices)):
            raise ValueError(f"Voice index must be between 0 and {len(voices)-1}")
        
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        self.engine.setProperty('voice', voices[voice_index].id)

    def speak(self, text: str) -> None:
        if not isinstance(text, str):
            raise ValueError(f"The provided text is not string!")
        self.engine.say((text))
        self.engine.runAndWait()

    def stop(self) -> None:
        self.engine.stop()
        
if __name__ == "__main__":
    _speaking = Text_To_Speech_Pyttsx3()
    _speaking.speak("Hello there my friend")
    _speaking.config_voice(165, 0.8) #the first number is for rate, second is the volume, and last is the voice_id
    try:
        _speaking.speak(56) #this would raise valueError, dont worry, the code is fine :)
    except ValueError as e:
        traceback.print_exc() 