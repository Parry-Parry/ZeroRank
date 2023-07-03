from typing import List, Union
import pandas as pd
from abc import abstractmethod
from numpy import array_split
import torch
import re
from .configs import FLAN_T5

clean = lambda x : re.sub(r"[^a-zA-Z0-9¿]+", " ", x)

class GenericModel:
    def __init__(self, 
                 generation_config : dict = None, 
                 num_return_sequences : int = None,
                 batch_size : int = 1, 
                 device = 'cpu') -> None:
        
        if not generation_config: generation_config = FLAN_T5
        if num_return_sequences: generation_config['num_return_sequences'] = num_return_sequences
        self.generation_config = generation_config
        self.batch_size = batch_size
        self.device = torch.device(device)

    @abstractmethod
    def logic(self, input : Union[str, pd.Series]) -> Union[str, List[str]]:
        raise NotImplementedError("This method must be implemented in a subclass")
    
    def generate(self, input : Union[str, pd.Series, pd.DataFrame]) -> Union[str, pd.Series]:
        if input is isinstance(input, str):
            return self.logic(input)
        elif isinstance(input, pd.Series):
            return pd.concat([self.logic(chunk.tolist()) for chunk in array_split(input, len(input) // self.batch_size)])
        else: 
            raise TypeError("Input must be a string or a pandas Object")

class FLANT5(GenericModel):
    def __init__(self, 
                 model_name : str, 
                 generation_config: dict = None, 
                 num_return_sequences: int = 1, 
                 batch_size: int = 1, 
                 device = 'cpu') -> None:
        super().__init__(generation_config, num_return_sequences, batch_size, device)

        from transformers import T5ForConditionalGeneration, T5TokenizerFast
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
    
    def postprocess(self, text):
        text = [clean(' '.join(t)) for t in text]
        return text if len(text) > 1 else text[0]

    def logic(self, input : Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(input, str): input = [input]

        inputs = self.tokenizer(input, padding = True, truncation = True, return_tensors = 'pt').to(self.device)
        outputs = self.model.generate(**inputs, **self.generation_config)
        outputs_text = self.tokenizer.batch_decode(outputs, skip_special_tokens = True)
        return self.postprocess(outputs_text)