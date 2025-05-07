import google.generativeai as genai #need pip install google-generativeai
import pyperclip #need pip install pyperclip
from dotenv import load_dotenv #need pip install python-dotenv
import os

class GeminiClient:
    def __init__(self, model_name : str ='models/gemini-2.0-flash-lite', api_key : str = None, memories_length : int = 4, limit_word_per_respond : int = 150):
        #check data type
        self.__enforce_type(memories_length, int, "memories_length")
        self.__enforce_type(limit_word_per_respond, int, "limit_word_per_respond")
        
        load_dotenv() # load if there is env file
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("API key was not provided or found in the 'GEMINI_API_KEY' environment variable.")
        
        #check data type before proceed further into connecting
        self.__enforce_type(model_name, str, "model_name")
        self.__enforce_type(self.api_key, str, "api_key")
        
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
        
        self.history = []
        self.max_length = memories_length
        self.word_prompt = f"Please remember to answer with less than {limit_word_per_respond} words."
    
    def ask(self, prompt : str) -> str:
        #not str, return...
        self.__enforce_type(prompt, str, "prompt")
        
        if not prompt:
            print("WARNING: Empty prompt provided, skipping API call.")
            return None
        
        try:
            response = self.model.generate_content(self.word_prompt + prompt)

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
        answer = self.ask(self.word_prompt + prompt)
        if answer:
            try:
                pyperclip.copy(answer)
                print("Copy to clipboard the answer")
            except Exception as e:
                print("WARNING: CANT COPY TO CLIPBOARD")
        return answer

    def ask_with_memories(self, prompt : str) -> str: # use if you want to store history to ask again (cap with memories_length)
        line = "Here's our previous conversation:\n"
        if self.history:
            for turn in self.history:
                line += f"User: {turn['question']}\nAI: {turn['answer']}\n"
        line += "Now, continue the conversation:\n"
        line += prompt
        answer = self.ask(line)
        self.__add_turn(prompt, answer)
        return answer
    
    def _clear_history(self):
        if self.history:
            self.history.clear()
    
    def __enforce_type(self, value, expected_type, arg_name):
        if not isinstance(value, expected_type):
            raise TypeError(f"Argument '{arg_name}' must be of type {expected_type.__name__}, but received {type(value).__name__}")
    
    def __add_turn(self, question, answer):
        self.__enforce_type(question, str, "question")
        self.__enforce_type(answer, str, answer)

        if len(self.history) >= self.max_length:
            self.history.pop(0)  # Remove the oldest turn
        self.history.append({"question": question, "answer": answer})
    
    
if __name__ == "__main__":
    _client = GeminiClient()
    _client.list_models()
    _answer = _client.ask("What land animal do you think is the best?")
    print(_answer)
    
    _answer = _client.ask_with_memories("My name is An")
    print(_answer)
    _answer = _client.ask_with_memories("Do you remember what we're talking about?")
    print(_answer)
    