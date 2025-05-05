import google.generativeai as genai #need pip install google-generativeai
import pyperclip #need pip install pyperclip
from dotenv import load_dotenv #need pip install python-dotenv
import os

class GeminiClient:
    def __init__(self, model_name : str ='models/gemini-2.0-flash-lite', api_key : str = None):
        #if not str, gonna return error
        
        
        load_dotenv() # load if there is env file
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("API key was not provided or found in the 'GEMINI_API_KEY' environment variable.")
        
        #check data type before proceed further into connecting
        self.__enforce_type(model_name, str, "model_name")
        self.__enforce_type(api_key, str, "api_key")
        
        try:
            genai.configure(api_key=self.api_key)
            print("Gemini API configured successfully.")
        except Exception as e:
            print(f"Error configuring Gemini API: {e}")
            raise

        self.model_name = model_name
        try:
            self.model = genai.GenerativeModel(self.model_name)
            print(f"Successfully initialized model: {self.model_name}") 
            print(f"Using endpoint: {self.model.model_name}") 
        except Exception as e:
            print(f"ERROR: Error initializing model {self.model_name}: {e}")
            raise
    
    def ask(self, prompt : str) -> str:
        #not str, return...
        self.__enforce_type(prompt, str, "prompt")
        
        if not prompt:
            print("WARNING: Empty prompt provided, skipping API call.")
            return None
        
        try:
            response = self.model.generate_content(prompt)

            if response.parts:
                answer = response.text
                print("INFO: Received response successfully.") # Replaced logging.info
                return answer
            else:
                print("No response received. This could be due to safety filters or other reasons.") 
                return ""

        except Exception as e:
            print(f"ERROR: An error occurred during the API call: {e}") # Replaced logging.error
            return None   
        
    def list_models(self) -> None:
        #List model available base on your api_key
        print("Retrieve available models...")
        try:
            print("-" * 40)
            for model in genai.list_models():
                # List model with 'generateContent' only
                if 'generateContent' in model.supported_generation_methods:
                    print(f"Model: {model.name}")
                    print(f"  Name: {model.display_name}")
                    print(f"  Description: {model.description}")
                    print("-" * 40)
        except Exception as e:
            print(f"Error retrieving: {e}")
   
    def ask_and_copy_to_clipboard(self, prompt : str) -> str:
        answer = self.ask(prompt)
        if answer:
            try:
                pyperclip.copy(answer)
                print("Copy to clipboard the answer")
            except Exception as e:
                print("WARNING: CANT COPY TO CLIPBOARD")
        return answer

    def __enforce_type(self, value, expected_type, arg_name):
        if not isinstance(value, expected_type):
            raise TypeError(f"Argument '{arg_name}' must be of type {expected_type.__name__}, but received {type(value).__name__}")
    
if __name__ == "__main__":
    _client = GeminiClient()
    _client.list_models()
    _client.ask("Hello, how are you?")
    