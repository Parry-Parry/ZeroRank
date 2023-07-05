from typing import Any

class Example:
    DEFAULT_TEXT = "Query: {query}\nDocument: {document}\nRelevance: {relevance}\n"
    def __init__(self, text=None) -> None:
        self.text = text if text else self.DEFAULT_TEXT
    
    def construct(self, **kwargs):
        return self.text.format(**kwargs)
    
    def __call__(self, inputs) -> Any:
        if isinstance(inputs, list):
            return [self.construct(**i) for i in inputs]
        return self.construct(**inputs)

class FewShotPrompt:
    DEFAULT_TEXT = "{instruction}\n{context}Query: {query}\nDocument: {docno}\nRelavance:"
    def __init__(self, instruction, text=None, example_constructor=None) -> None:
        self.text = text.format(instruction=instruction) if text else self.DEFAULT_TEXT.format(instruction=instruction)
        self.example_constructor = example_constructor if example_constructor else Example()

    def example(self, examples):
        return self.example_constructor(examples)

    def construct(self, **kwargs):
        return self.text.format(**kwargs)

    def __call__(self, inputs) -> Any:
        if isinstance(inputs, list):
            return [self.construct(**i) for i in inputs]
        return self.construct(**inputs)