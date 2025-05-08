from gtts import gTTS, lang # need pip install gTTS
import time
import os
import playsound # need pip install playsound==1.2.2 (the later version wouldn't work)

def gtts_speak(text: str, lang: str ='vi') -> str:
    temp_filename = str(round(time.time() * 10)) + ".mp3"
    # Get the absolute path to the directory where the script is running
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Or use os.getcwd() if appropriate
    temp_ID_path = os.path.join(script_dir, temp_filename)

    tts = gTTS(text=text, lang=lang)
    tts.save(temp_ID_path) # change to save with full path

    try:
        playsound.playsound(temp_ID_path) # play mp3 file using full path
    finally:
        if os.path.exists(temp_ID_path): # delete mp3 file
            os.remove(temp_ID_path)
    return temp_ID_path # return full path 

#for those who need what language it supports and the code lang for it
def gtts_print_supported_languages() -> None:
    supported_languages = lang.tts_langs()  # Get the dictionary of supported languages
    print("Supported languages by gTTS:")
    for code, language in supported_languages.items():
        print(f"{code}: {language}")

#how to use
if __name__=="__main__":
    gtts_print_supported_languages()
    _temp_ID = gtts_speak("Xin chào người đẹp, em tên là gì thế", "vi")
    print("You can get file name as output: "+ _temp_ID)