# Contains standalone scripts for quick-support tasks; not for modular reuse.

from .geminiAPI import GeminiClient
from .cleaner import Cleaner
import pyautogui
import keyboard
import time
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define the full path for the screenshot
screenshot_path = os.path.join(script_dir, "screenshot.png")

def screenshot_ask_and_answer_clip(secret_key_for_gemini : str = None, hotkey_asking : str = "`"):
    search = False
    def screenshot():
        nonlocal search
        pyautogui.screenshot(screenshot_path)
        search = True
        
    if secret_key_for_gemini is None:
        print("Require secret key for gemini api")
        return
    
    client = GeminiClient(api_key=secret_key_for_gemini)
    keyboard.add_hotkey(hotkey_asking, screenshot)
    while True:
        if search:
            client.ask_and_copy_to_clipboard("Help me solve these questions please. Just tell me the answer, no need for explanation.", screenshot_path)
            search = False
            cleaner = Cleaner()
            cleaner.remove_all_files_end_with(".png")
        time.sleep(0.2)