# Library for people who want to do ocr but don't want to download Tesseract OCR
# Weaker than Tesseract OCR

import pyautogui # need pip install pyautogui (use to screenshot)
import easyocr# need pip install easyocr (use to read text from image)

class Text_Extractor_EasyOCR:
    def __init__(self, language : str ='en') -> None:
        #check type of the input
        self.__enforce_type(language, str, "language")
        
        self.reader = easyocr.Reader([language])
        self.screen_width, self.screen_height = pyautogui.size()

    def get_text_from_screen(self, capture_region: tuple = None) -> str:
        if capture_region is None: #if default, take fullscreen
            capture_region = (0, 0, self.screen_width, self.screen_height)
            
        #check type of the input
        self.__enforce_type(capture_region, tuple, "capture_region")
            
        screenshot = pyautogui.screenshot(region=capture_region)  #capture
        screenshot.save("screenshot.png") #careful if there is a file screenshot before run, it would replace that file
        result = self.reader.readtext("screenshot.png", detail=0)
        return " ".join(result)  #extract and return str

    def set_capture_region(self, crop_left: float = 0, crop_right: float = 0, crop_up: float = 0, crop_down: float = 0) -> tuple:
        #check type of the input
        self.__enforce_type(crop_left, (int, float), "crop_left")
        self.__enforce_type(crop_right, (int, float), "crop_right")
        self.__enforce_type(crop_up, (int, float), "crop_up")
        self.__enforce_type(crop_down, (int, float), "crop_down")
        
         # Validate crop percentage values
        if not (0 <= crop_left <= 100):
            raise ValueError("crop_left must be between 0 and 100")
        if not (0 <= crop_right <= 100):
            raise ValueError("crop_right must be between 0 and 100")
        if not (0 <= crop_up <= 100):
            raise ValueError("crop_up must be between 0 and 100")
        if not (0 <= crop_down <= 100):
            raise ValueError("crop_down must be between 0 and 100")
        
        top_percentage_to_ignore = crop_up / 100
        bottom_percentage_to_ignore = crop_down / 100
        left_percentage_to_ignore = crop_left / 100
        right_percentage_to_ignore = crop_right / 100

        # Height of region to capture
        top_y = int(self.screen_height * top_percentage_to_ignore)
        bottom_y = int(self.screen_height * (1 - bottom_percentage_to_ignore))
        capture_height = bottom_y - top_y

        # The width of the region to capture
        left_x = int(self.screen_width * left_percentage_to_ignore)
        right_x = int(self.screen_width * (1 - right_percentage_to_ignore))
        capture_width = right_x - left_x 

        return (left_x, top_y, capture_width, capture_height)
        
    def __enforce_type(self, value, expected_types, arg_name):
        if not isinstance(value, expected_types):
            expected_names = [t.__name__ for t in expected_types] if isinstance(expected_types, tuple) else [expected_types.__name__]
            expected_str = ", ".join(expected_names)
            raise TypeError(f"Argument '{arg_name}' must be of type {expected_str}, but received {type(value).__name__}")
        
# Example usage:
if __name__ == "__main__":
    _extractor = Text_Extractor_EasyOCR(language='en')
    # Extract text from the screen
    _text = _extractor.get_text_from_screen()
    print(f"Extracted text: {_text}")
    
    # Set capture region (optional, defaults to full screen if not specified)
    _region = _extractor.set_capture_region(crop_left=20, crop_right=5, crop_up=20, crop_down=5)
    print(f"Capture region: {_region}")

    # Extract text from the screen
    _text = _extractor.get_text_from_screen(capture_region=_region)
    print(f"Extracted text: {_text}")

#SUPPORTED LANG easyocr:
# Abaza	abq
# Adyghe	ady
# Afrikaans	af
# Angika	ang
# Arabic	ar
# Assamese	as
# Avar	ava
# Azerbaijani	az
# Belarusian	be
# Bulgarian	bg
# Bihari	bh
# Bhojpuri	bho
# Bengali	bn
# Bosnian	bs
# Simplified Chinese	ch_sim
# Traditional Chinese	ch_tra
# Chechen	che
# Czech	cs
# Welsh	cy
# Danish	da
# Dargwa	dar
# German	de
# English	en
# Spanish	es
# Estonian	et
# Persian (Farsi)	fa
# French	fr
# Irish	ga
# Goan Konkani	gom
# Hindi	hi
# Croatian	hr
# Hungarian	hu
# Indonesian	id
# Ingush	inh
# Icelandic	is
# Italian	it
# Japanese	ja
# Kabardian	kbd
# Kannada	kn
# Korean	ko
# Kurdish	ku
# Latin	la
# Lak	lbe
# Lezghian	lez
# Lithuanian	lt
# Latvian	lv
# Magahi	mah
# Maithili	mai
# Maori	mi
# Mongolian	mn
# Marathi	mr
# Malay	ms
# Maltese	mt
# Nepali	ne
# Newari	new
# Dutch	nl
# Norwegian	no
# Occitan	oc
# Pali	pi
# Polish	pl
# Portuguese	pt
# Romanian	ro
# Russian	ru
# Serbian (cyrillic)	rs_cyrillic
# Serbian (latin)	rs_latin
# Nagpuri	sck
# Slovak	sk
# Slovenian	sl
# Albanian	sq
# Swedish	sv
# Swahili	sw
# Tamil	ta
# Tabassaran	tab
# Telugu	te
# Thai	th
# Tajik	tjk
# Tagalog	tl
# Turkish	tr
# Uyghur	ug
# Ukranian	uk
# Urdu	ur
# Uzbek	uz
# Vietnamese	vi