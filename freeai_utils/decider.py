class DecisionMaker:
    #this class should only answer in 2 ways only
    def __init__(self, sample_ques_ans : str = None, positive_ans = "YES", negative_ans = "NO", model : str = "google/flan-t5-base", device : str = None, generation_kwargs : dict = None) -> None:
        pass

    def decide(self, prompt : str = None) -> str:
        pass
    
    def _run_pre_built_sample(self) -> None:
        pass