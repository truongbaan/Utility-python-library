import pytest
import gc
from freeai_utils.decider import DecisionMaker

@pytest.fixture(scope="module")
def decision_model():
    model = DecisionMaker()
    yield model
    del model
    gc.collect()

def test_model_initialized(decision_model):
    assert decision_model.model is not None
    assert decision_model.model_name is not None
    assert decision_model.device in ("cpu", "cuda")
    assert decision_model.tokenizer is not None
    assert decision_model.system_prompt is not None
    
    #test modification 
    with pytest.raises(AttributeError, match="Cannot reassign '_model_name' after initialization"):
        decision_model._model_name = None
    with pytest.raises(AttributeError, match="Cannot reassign '_model' after initialization"):
        decision_model._model = None
    with pytest.raises(AttributeError, match="Cannot reassign '_device' after initialization"):
        decision_model._device = "cpu"
    with pytest.raises(AttributeError, match="Cannot reassign '_tokenizer' after initialization"):
        decision_model._tokenizer = "hi"
    
    #check prompt modification
    decision_model.construct_sys_prompt("", "Yes", "No")
    cons_prompt = (
            "Analyze the following question and determine if an internet search is required to answer it. "
            f"Respond with 'Yes' or 'No'.\n\n"
            f"Example:\n<no example provided>"
        )
    print(f"Real: {decision_model._system_prompt}")
    print(f"Expected: {cons_prompt}")
    
    assert decision_model._system_prompt == cons_prompt

def test_func(decision_model):
    example_questions = [
            "What is the capital of Germany?", #search
            "Whats the latest stock price of Tesla?", #search
            "Who won the World Cup in 2018?",#search
            "What is the weather like in London tomorrow?", #search
            "What is the capital of France", #search
            "What was the score of the latest football match between Manchester United and Arsenal", #search
            "What is photosynthesis?", #search
            "Who is the current President of the United States", #search
            "When was the Earth formed", #no
            "How to use Python 3.12",#search
            "What is the current time in Tokyo", #search
            "Who invented the light bulb" #search
        ]
    example_ans = [
        "SEARCH_INTERNET",
        "SEARCH_INTERNET",
        "SEARCH_INTERNET",
        "SEARCH_INTERNET",
        "SEARCH_INTERNET",
        "SEARCH_INTERNET",
        "SEARCH_INTERNET",
        "SEARCH_INTERNET",
        "NO_SEARCH_NEEDED",
        "SEARCH_INTERNET",
        "SEARCH_INTERNET",
        "SEARCH_INTERNET"
    ]
    decision_model.config_default_internet_search() #guide for model
    for idx, ques in enumerate(example_questions):
        assert decision_model.decide(ques) == example_ans[idx]