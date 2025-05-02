# Library for people who want to do ocr but don't want to download Tesseract OCR
# Weaker than Tesseract OCR

import pyautogui # need pip install pyautogui (use to screenshot)
import keyboard #need pip install keyboard (use to run function with hotkey)
import easyocr# need pip install easyocr (use to read text from image)
import pyperclip #need pip install pyperclip (use to copy to clipboard)
from wrapper import time_it

""" Copy part """
reader = easyocr.Reader(['en'])

@time_it
def get_text_from_screen() -> str: #func to get text 
    screenshot = pyautogui.screenshot()
    screenshot.save("screenshot.png")
    result = reader.readtext("screenshot.png", detail=0)
    for word in result:
        print(word, end = " ")
    return " ".join(result)

def on_key(): #func to just copy to clipboard
    print("Key detected. Processing...")
    text = get_text_from_screen()
    if text and text.strip():
        pyperclip.copy(text)
    print("Text from screen:", text)

""" Copy part """

if __name__ == "__main__":
    # Run when "a" key is pressed``
    keyboard.add_hotkey("`", on_key)

    print("Running... Press 'a' to screenshot and auto-answer.")
    keyboard.wait("esc")
