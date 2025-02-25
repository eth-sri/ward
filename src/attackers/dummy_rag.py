import random
from typing import Any, Dict, List, Optional, Sequence, Tuple


class DummyRag:

    def __init__(self, model: Any, contexts: List[Tuple[str, str]]):
        self.model = model
        self.contexts = {}
        self.contexts_loose = {}
        for i, (idd, doc) in enumerate(contexts):
            self.contexts[idd] = doc
            self.contexts_loose[idd.split("_")[0]] = doc

    def generate(self, prompts: List[str], explicit_ids: List[str]) -> Any:
        full_prompts = []
        for p, idd in zip(prompts, explicit_ids):
            if idd in self.contexts:
                context = self.contexts[idd]
            elif idd in self.contexts_loose:
                context = self.contexts_loose[idd]
            else:
                context = random.choice(list(self.contexts.values()))

            full_prompts.append(f"Answer based on the context: {context}. Question: {p}")
        responses = []
        batch_size = 20
        for i in range(0, len(full_prompts), batch_size):
            print(f"Batch {i} to {min(len(full_prompts), i + batch_size)}")
            batch = full_prompts[i : min(len(full_prompts), i + batch_size)]
            responses += self.model.generate(batch)[0]
        return responses
