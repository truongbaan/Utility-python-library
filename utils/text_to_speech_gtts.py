from gtts import gTTS, lang # need pip install gTTS
import time
import os
import playsound # need pip install playsound

""" Copy part """

def gtts_speak(text: str, lang: str ='vi') -> str: #return 'temp_ID.mp3' for deleting manually
    temp_ID = (str)(round(time.time() * 10)) + ".mp3"
    tts = gTTS(text=text, lang=lang)
    tts.save(temp_ID) # create mp3 file
    playsound.playsound(temp_ID) # play mp3 file
    if os.path.exists(temp_ID): # delete mp3 file
        os.remove(temp_ID)
    return temp_ID #this is for interruption of the function when speaking, if needed, you can manually delete the file 

""" Copy part """

#for those who need what language it supports and the code lang for it
def gtts_print_supported_languages():
    supported_languages = lang.tts_langs()  # Get the dictionary of supported languages
    print("Supported languages by gTTS:")
    for code, language in supported_languages.items():
        print(f"{code}: {language}")

#how to use
if __name__=="__main__":
    gtts_print_supported_languages()
    temp_ID = gtts_speak("Xin chào người đẹp, em tên là gì thế", "vi")
    print("You can get file name as output: "+ temp_ID)