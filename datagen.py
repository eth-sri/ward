import json
import random

from datasets import load_dataset

from src.models.api_model import APIModel


def datagen() -> None:
    print("Generating data")
    out_dir = "farad/"

    gpt4 = APIModel("gpt4o")
    claude = APIModel("claude3.5sonnet")
    llama_big = APIModel("llama3.1-405b")
    qwen = APIModel("qwen1.5-110b")
    models = [gpt4, claude, llama_big, qwen]

    # Load the dataset and prepare to extract key facts
    ds = load_dataset("ServiceNow/repliqa")["repliqa_0"]
    it = 0
    data = []
    for idx in range(3591):
        doc = ds[it]["document_extracted"]
        topic = ds[it]["document_topic"]
        it += 5
        #skip
        #if idx < 3391:
        #    continue
        data.append({"id": f"{idx:04}", "topic": topic, "original_doc": doc})

    # Fact extraction
    sysprompt = f"You are a chatbot that extracts facts from documents. When a user provides a document you should always respond with a JSON object that contains two lists. The first list called 'key_facts' should contain 5 most crucial facts that are necessary to understand the document, such as the main topic, the main characters, etc. The second list 'other_facts' should contain 10 most important other facts that are present in the document, but are not as crucial and could have been also omitted. Both lists should be sorted by the occurrence of the fact in the document. Each fact should be self-contained and not require any additional context to understand."
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "schema": {
                "type": "object",
                "properties": {
                    "key_facts": {"type": "array", "items": {"type": "string"}},
                    "other_facts": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["key_facts", "other_facts"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
    prompts = []
    for idx, d in enumerate(data):
        prompts.append(f"Document: {d['original_doc']}")
    responses = gpt4.generate(sysprompt=sysprompt, prompts=prompts, response_format=response_format)
    responses_dicts = [json.loads(r) for r in responses]
    for idx, d in enumerate(data):
        d["key_facts"] = responses_dicts[idx]["key_facts"]
        d["other_facts"] = responses_dicts[idx]["other_facts"]
        d["articles"] = {}
        with open(f"{out_dir}/{d['id']}.json", "w") as f:
            json.dump(d, f, indent=4)

    print("Fact extraction done, going forward.")
    for i, model in enumerate(models):
        # Now generate the actual article
        sysprompt = f"You are a chatbot that writes articles. The user will provide you with a list of facts. Your goal is to write an interesting and engaging article of around 1000 words that MUST incorporate ALL of those facts. Always output AT LEAST 500 WORDS. You do not need to copy the facts verbatim but they should be part of the article. Feel free to be creative in how you piece the facts together. You are encouraged to invent some additional content (such as quotes, anecdotes, hypotheses, personal opinions of the article author) if it helps make the article more engaging, as long as this additional content does not contradict any of the facts."
        facts = []
        for idx, d in enumerate(data):
            curr_facts = list(d["key_facts"])
            random.shuffle(d["other_facts"])
            curr_facts.extend(d["other_facts"][:2])
            facts.append(curr_facts)
        prompts = [f"FACTS: {str(f)}" for f in facts]
        responses = model.generate(sysprompt=sysprompt, prompts=prompts)
        for idx, (response, d) in enumerate(zip(responses, data)):
            d["articles"][model.model_name] = {
                "facts": facts[idx],
                "article": response,
                "num_words": len(response.split()),
            }
            with open(f"{out_dir}/{d['id']}.json", "w") as f:
                json.dump(d, f, indent=4)

        print(f"Done with {model}.")


# go 
datagen()
