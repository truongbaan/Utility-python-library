# Library for people who want to do ocr but don't want to download Tesseract OCR
# Weaker than Tesseract OCR

import pyautogui # need pip install pyautogui (use to screenshot)
import easyocr# need pip install easyocr (use to read text from image)

""" Copy part """

reader = easyocr.Reader(['en'])
screen_width, screen_height = pyautogui.size()

def get_text_from_screen(capture_region = (0,0,screen_width,screen_height)) -> str: #func to get text 
    screenshot = pyautogui.screenshot(region = capture_region) #select region if want, default is capture fullscreen
    screenshot.save("screenshot.png")
    result = reader.readtext("screenshot.png", detail=0)
    # for word in result:
    #     print(word, end = " ")
    return " ".join(result)

""" Copy part """

#how to use
if __name__ == "__main__":
    text = get_text_from_screen()
    print("Text from screen:", text)
    print("Running... Press 'a' to screenshot and auto-answer.")
