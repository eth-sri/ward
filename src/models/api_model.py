import os
import re
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import partial
from typing import Any, Iterator, List, Tuple

import anthropic
import openai
import together
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, LogitsProcessorList, PreTrainedTokenizer

from src.config import MetaConfig, ModelConfig
from src.models.utils import LogitInfo
from src.utils import print

full_model_names = {
    "gpt3.5": "gpt-3.5-turbo-0125",
    "gpt4o": "gpt-4o-2024-08-06",
    "claude3.5sonnet": "claude-3-5-sonnet-20240620",
    "claude3haiku": "claude-3-haiku-20240307",
    "llama3.1-405b": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "qwen1.5-110b": "Qwen/Qwen1.5-110B-Chat",
    "llama3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "llama3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
}


class APIModel:
    def __init__(self, model_name: str) -> None:
        if model_name in ["gpt4o", "gpt3.5"]:
            self.provider = "openai"
            self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        elif model_name in ["claude3.5sonnet", "claude3haiku"]:
            self.provider = "anthropic"
            self.client = anthropic.Anthropic()  # ANTHROPIC_API_KEY
        elif model_name in ["llama3.1-405b", "qwen1.5-110b", "llama3.1-8b", "llama3.1-70b"]:
            self.provider = "together"
            self.client = together.Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        else:
            raise RuntimeError(f"Unknown model name: {model_name}")
        self.model_name = model_name
        self.full_model_name = full_model_names[model_name]

    def _query_api(self, sysprompt: str, ID: int, prompt: str, response_format: Any = None) -> str:
        if response_format is not None and self.provider != "openai":
            raise RuntimeError("response_format is only supported for OpenAI models")
        try:
            if self.provider == "openai":
                payload = [
                    {"role": "system", "content": sysprompt},
                    {"role": "user", "content": prompt},
                ]
                request = partial(
                    self.client.chat.completions.create,
                    model=self.full_model_name,
                    messages=payload,
                    temperature=1,
                    max_tokens=4096,
                )
                if response_format is None:
                    response = request().choices[0].message.content
                else:
                    response = request(response_format=response_format).choices[0].message.content
            elif self.provider == "anthropic":
                payload = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]  # type: ignore
                response = (
                    self.client.messages.create(
                        model=self.full_model_name,
                        system=sysprompt,
                        messages=payload,
                        temperature=1,
                        max_tokens=4096,
                    )
                    .content[0]
                    .text
                )
            elif self.provider == "together":
                payload = [
                    {"role": "system", "content": sysprompt},
                    {"role": "user", "content": prompt},
                ]
                response = (
                    self.client.chat.completions.create(
                        model=self.full_model_name, messages=payload, temperature=1, max_tokens=4096
                    )
                    .choices[0]
                    .message.content
                )
            else:
                raise RuntimeError(f"Unknown provider: {self.provider}")
        except Exception as e: 
            # we need to know on which ID we crashed
            e.add_note(str(ID))
            raise e 
        return response

    def generate(
        self, sysprompt: str, prompts: List[str], response_format: Any = None
    ) -> List[str]:
        max_workers = 8
        base_timeout = 240
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            ids_to_do = list(range(len(prompts)))
            retry_ctr = 0
            timeout = base_timeout

            while len(ids_to_do) > 0 and retry_ctr <= len(prompts):
                results = executor.map(
                    lambda id: (
                        id,
                        prompts[id],
                        self._query_api(sysprompt, id, prompts[id], response_format=response_format),
                    ),
                    ids_to_do,
                    timeout=timeout,
                )
                try:
                    for res in tqdm(
                        results,
                        total=len(ids_to_do),
                        desc="Queries",
                        position=1,
                        leave=False,
                    ):
                        id, orig, answer = res
                        yield answer
                        # answered_prompts.append()
                        ids_to_do.remove(id)
                except TimeoutError:
                    print(f"Timeout: {len(ids_to_do)} prompts remaining")
                except openai.RateLimitError as r:
                    print(f"Rate Limit: {r}")
                    time.sleep(10)
                    continue
                except Exception as e:
                    print(f"Exception: {e}")
                    if getattr(e, "type", "default") == "invalid_prompt" and len(ids_to_do) == 1:
                        # We skip invalid prompts for GPT -> This only works with bs=1
                        ids_to_do = []
                        yield "invalid_prompt_error"
                    elif "maximum context length" in e.message:
                        # how much are we over?
                        pattern = r"(\d+) tokens. However, you requested (\d+)"
                        matches = re.findall(pattern, e.message)
                        max_len, curr_len = matches[0] 
                        chars_to_cut = max((int(curr_len) - int(max_len)) * 4, 400) # estimate but never below 400 (~100 toks)

                        #  presumably the first ID is the one that crashed?
                        curr_id = int(e.__notes__[0])
                        assert curr_id in ids_to_do

                        # split and shorten
                        toks = prompts[curr_id].split("</documents>")
                        if len(toks) == 2:
                            toks[0] = toks[0][:-chars_to_cut] # crop last 1000
                            prompts[curr_id] = f"{toks[0]}</documents>{toks[1]}"
                            print(f"Shortened prompt {curr_id} by {chars_to_cut} chars.")
                        else:
                            print(f"Tried to shorten the prompt but there was an error....Got {len(toks)} toks.")
                        time.sleep(3)
                    else:
                        time.sleep(10)
                    continue

                if len(ids_to_do) == 0:
                    break

                time.sleep(2 * retry_ctr)
                timeout *= 2
                timeout = min(base_timeout, timeout)
                retry_ctr += 1
