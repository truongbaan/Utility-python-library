from langdetect import detect, detect_langs #need pip install langdetect
from typing import Union, Tuple
from googletrans import Translator # pip install googletrans==4.0.0rc1
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, MBart50TokenizerFast, MBartForConditionalGeneration

class LangTranslator:
    def __init__(self, local_status : str = "backup", local_model_num : int = 1): #not done
        self.online_translator = Translator()
        self.local_translator = None
        if local_status not in ["active", "inactive", "backup"]:
            raise ValueError(f"local_status could only be \'active\', \'inactive\', \'backup\'. Current value: {local_status}")
        
        if local_status in ["active", "backup"]:
            self.local_translator = LocalTranslator(local_model_num)
        
        #local_status between active, inactive, backup where active will pioritize, inactive will unable it, backup will enable and use when online fail
        pass

    def translate(self, text_to_translate, tgt_lang = 'en', src_lang = 'auto'): #not done
        return self.online_translator.translate(text_to_translate, dest=tgt_lang, src=src_lang).text
    
    def detect_language(self, prompt) -> Tuple[str, float]: #completed
        detected_language = detect_langs(prompt)

        if detected_language:
            return detected_language[0].lang, detected_language[0].prob
        else: return "unknown", 0.0
    
class LocalTranslator: #not done
    def __init__(self, local_model_num : int = 1):
        _amount_translator = _Class_Counter().get_count() - 3
        self.model = None
        if local_model_num > _amount_translator or local_model_num < 1:
            raise ValueError(f"local_model_num could only between 1 and {_amount_translator}")
        self.__init_local_translator(local_model_num)

    def translate(self, prompt : str, src_lang : str, dest_lang : str):
        self.tokenizer.src_lang = src_lang
        encoded_hi = self.tokenizer(prompt, return_tensors="pt")
        generated_tokens = self.model.generate(**encoded_hi, forced_bos_token_id=self.tokenizer.get_lang_id(dest_lang))
        print(self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
        
    def __init_local_translator(self, num : int):
        match num:
            case 1:
                self.model = "Choosing models..."
            case 2:
                self.model = "Choosing models..."
        pass    
    
######################################### Use only for providing number for possible local model Class
import inspect
import sys
import ast
import os

class _Class_Counter:
    def __init__(self):
        self.class_count = 0
        self.classes = []

        file_path = None
        try:
            file_path = inspect.getsourcefile(self.__class__)
            if file_path is None:
                module = sys.modules.get(self.__class__.__module__)
                if module and hasattr(module, '__file__'):
                    file_path = module.__file__

            if file_path:
                file_path = os.path.abspath(file_path)

        except Exception as e:
            print(f"Error determining source file for Counter class: {e}")
            return
        if not file_path:
            print("Could not determine the source file for this class. Cannot count classes.")
            return
        print(f"Counting classes in the file where Counter is defined: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()
            tree = ast.parse(source_code)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self.class_count += 1
                    self.classes.append(node.name)

        except FileNotFoundError:
            print(f"Error: Source file '{file_path}' not found during AST parsing.")
            self.class_count = 0
            self.classes = []
        except Exception as e:
            print(f"Error parsing AST of file '{file_path}': {e}")
            self.class_count = 0
            self.classes = []

    def get_count(self):
        return self.class_count

    def get_class_names(self):
        return self.classes

if __name__ == "__main__":
    text3 = "Đây là một đoạn văn bản mẫu bằng tiếng Việt."
    trans = LangTranslator()
    print(trans.detect_language(text3))
    print(trans.translate(text3))