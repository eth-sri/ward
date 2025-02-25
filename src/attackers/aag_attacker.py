import json
import random
from typing import Any, Dict, List, Optional, Tuple

from src.attackers.dummy_rag import DummyRag
from src.config import AttackerConfig, MetaConfig
from src.models import APIModel, HfModel
from src.rag_system import RagSystem


class AagAttacker:

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
        self.give_explicit_ids = attacker_cfg.give_explicit_ids

        # Trained
        self.threshold = 0.495
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
        score_yeses = sum([1 for k, v in signals_yeses.items() if v["yes"] == 1]) / len(
            signals_yeses
        )
        signals_noes = self._get_signals(rag, a4)
        score_noes = sum([1 for k, v in signals_noes.items() if v["yes"] == 1]) / len(signals_noes)
        print(f"yeses: {score_yeses}, noes: {score_noes}")
        threshold = (score_yeses + score_noes) / 2
        print(f"threshold: {threshold}")
        print("Done training.")

    def register_corpus(self, corpus: List[Tuple[str, str]]) -> None:
        self.corpus = corpus #  # list of (id, doc) tuples

    def get_corpus_for_rag(self) -> List[Tuple[str, str]]:
        return self.corpus

    def infer_membership(self, rag_system: RagSystem) -> Tuple[bool, float]:
        signals = self._get_signals(rag_system, self.corpus)

        good, total = 0, 0
        bitstring = ""
        for idd, signal in signals.items():
            if signal["yes"] == 1:
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
        # Build RAG queries
        rag_queries = []
        explicit_ids = []
        for idd, doc in targets:
            rag_queries.append(
                f"Does this:\n'{doc[:1000]}'\nappear in the context? Answer with Yes or No."
            )
            explicit_ids.append(idd)
        # Query RAG and get the score
        if not self.give_explicit_ids:
            explicit_ids = None

        rag_responses = rag_system.generate(rag_queries, explicit_ids=explicit_ids)
        signals = {}
        for idd, resp, q, doc in zip(explicit_ids, rag_responses, rag_queries, targets):
            if "yes" in resp[:10].lower():
                signals[idd] = {"yes": 1}
            else:
                signals[idd] = {"yes": 0}

        return signals
