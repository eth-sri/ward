import json
import random
from typing import Any, Dict, List, Optional, Tuple

from src.attackers.dummy_rag import DummyRag
from src.config import AttackerConfig, MetaConfig
from src.models import APIModel, HfModel
from src.rag_system import RagSystem


class FactsAttacker:

    def __init__(
        self,
        meta_cfg: MetaConfig,
        attacker_cfg: AttackerConfig,
    ) -> None:
        self.seed, self.device, self.out_root_dir = (
            meta_cfg.seed,
            meta_cfg.device,
            meta_cfg.out_root_dir,
        )
        self.meta_cfg = meta_cfg
        self.cfg = attacker_cfg
        self.corpus: List[Tuple[str, str]] = []
        self.gpt = APIModel("gpt4o")
        self.give_explicit_ids = attacker_cfg.give_explicit_ids

        # Trained
        self.threshold = 0.505  # 0.02 vs 0.99
        pass

    def train(self, trainset: List[Dict[str, Any]]) -> None:
        a1, a4 = [], []
        random.Random(123).shuffle(trainset)
        for i, d in enumerate(trainset):
            if i < 100:
                a1.append((f'{d["id"]}_gpt4o', d["articles"]["gpt4o"]["article"]))
            else:
                a4.append((f'{d["id"]}_qwen1.5-110b', d["articles"]["qwen1.5-110b"]["article"]))
        assert len(a1) == 100
        assert len(a4) == 100

        tmp_model = HfModel(self.meta_cfg, self.cfg.model)
        rag = DummyRag(tmp_model, a1)
        signals_yeses = self._get_signals(rag, a1)
        score_yeses = sum([1 for k, v in signals_yeses.items() if v["unanswerable"] == 0]) / len(
            signals_yeses
        )
        signals_noes = self._get_signals(rag, a4)
        score_noes = sum([1 for k, v in signals_noes.items() if v["unanswerable"] == 0]) / len(
            signals_noes
        )
        print(f"yeses: {score_yeses}, noes: {score_noes}")
        threshold = (score_yeses + score_noes) / 2
        print(f"threshold: {threshold}")
        print("Done training.")

    def register_corpus(self, corpus: List[Tuple[str, str]]) -> None:
        self.corpus = corpus  # list of (id, doc) tuples

    def get_corpus_for_rag(self) -> List[Tuple[str, str]]:
        return self.corpus

    def infer_membership(self, rag_system: RagSystem) -> Tuple[bool, float, Any]:
        signals = self._get_signals(rag_system, self.corpus)

        good, total = 0, 0
        bitstring = ""
        for idd, signal in signals.items():
            if signal["unanswerable"] == 0:
                good += 1
                bitstring += "1"
            else:
                bitstring += "0"
            total += 1
        score = good / total

        return score > self.threshold, score, bitstring

    def _get_signals(
        self, rag_system: RagSystem | DummyRag, targets: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        # TODO: cache needs to take into account which of the 4 models generated
        with open("cache/facts_attacker.json", "r") as f:
            cache = json.load(f)

        # Sometimes we need to query GPT to generate questions, some are already cached
        data = {}
        gpt_queries = []
        sysprompt = "For a given document, generate a single simple question that can only be answered by reading the document, and is absolutely unanswerable for someone who has not read the document."
        for idd, doc in targets:
            data[idd] = {"doc": doc}
            # find a question that can only be answered with the doc?
            # ask to answer truthfully and check for Unanswerable
            if idd not in cache:
                gpt_queries.append((idd, f"Document: {doc}"))
            else:
                data[idd]["question"] = cache[idd]

        # Parse GPT responses and save to cache
        gpt_responses = self.gpt.generate(sysprompt, [q[1] for q in gpt_queries])
        for (idd, _), resp in zip(gpt_queries, gpt_responses):
            data[idd]["question"] = resp
            cache[idd] = resp
        with open("cache/facts_attacker.json", "w") as f:
            json.dump(cache, f, indent=4)
        print(f"Saved cache to cache/facts_attacker.json")

        # Build RAG queries
        rag_queries = []
        explicit_ids = []
        for idd in data.keys():
            rag_queries.append(
                f"{data[idd]['question']}\nDo not hallucinate the answer. If you can answer start your response immmediatelly, otherwise, if you can not answer truthfully, it is very important to start the response with the word 'Unanswerable:'. Feel free to explain why it is unanswerable after, but always start such responses with the word Unanswerable."
            )
            explicit_ids.append(idd)

        # Query RAG and get the score
        if not self.give_explicit_ids:
            explicit_ids = None
        rag_responses = rag_system.generate(rag_queries, explicit_ids=explicit_ids)
        signals = {}
        for idd, resp, q in zip(explicit_ids, rag_responses, rag_queries):
            if "Unanswerable" in resp or "unanswerable" in resp:
                signals[idd] = {"unanswerable": 1}
            else:
                signals[idd] = {"unanswerable": 0}
        return signals
