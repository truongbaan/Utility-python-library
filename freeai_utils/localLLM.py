from transformers import AutoModelForCausalLM, AutoTokenizer
from freeai_utils.log_set_up import setup_logging
import logging
from typing import Union
import torch

#other model: Qwen/Qwen3-4B, or any models that is Qwen
class LocalLLM:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", preferred_device: str = "cuda", memories_length: int = 4):
        self.device = preferred_device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=self.device
        ).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._history = []
        self._max_length = memories_length
        
    def ask(self, messages: list[dict]):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=32768
            )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
    
        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        return content, thinking_content
        
    def ask_with_memories(self, prompt: str) -> tuple[str, str]:
        messages = []
        # for turn in self._history:
        #     messages.append({"role": "user", "content": turn["question"]})
        #     messages.append({"role": "system", "content": turn["answer"]})
        line = "Here's our previous conversation:\n"
        if self._history:
            for turn in self._history:
                line += f"User: {turn['question']}\nAI: {turn['answer']}\n"
        line += "Now, continue the conversation:\n"
        line += prompt
        messages.append({"role": "user", "content": line})

        content, thinking_content = self.ask(messages)
        self.__add_turn(prompt, content)
        return content, thinking_content
        
    def _clear_history(self) -> None:
        if self._history:
            self._history.clear()
    
    def __add_turn(self, question, answer) -> None:
        self.__enforce_type(question, str, "question")
        self.__enforce_type(answer, str, "answer")
    
        while len(self._history) >= self._max_length:
            self._history.pop(0)
        self._history.append({"question": question, "answer": answer})
    
    def __enforce_type(self, value, expected_type, arg_name):
        if not isinstance(value, expected_type):
            raise TypeError(f"Argument '{arg_name}' must be of type {expected_type.__name__}, but received {type(value).__name__}")
    
if __name__ == "__main__":
    lm = LocalLLM(preferred_device="cpu")
    print(lm.ask([{"role": "user", "content": "Hi, how are you? When will you use thinking mode and when will not?"}]))
    print(lm.ask_with_memories("Hi, my name is Andy, what is your favorite animal")[0])
    print(lm.ask_with_memories("Hi, do you remember our conversation,could you tell me about it?")[0])