import json
import random
from typing import Any, Dict, List, Optional, Tuple

from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

from src.attackers.dummy_rag import DummyRag
from src.config import AttackerConfig, MetaConfig
from src.models import APIModel, HfModel
from src.rag_system import RagSystem


class SibAttacker:

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
        self.threshold = 0.51
        self.sim_thresh, self.ppl_thresh = 0.75521, 14.46141

    def set_models(self, aux_model: HfModel, sim_model: SentenceTransformer) -> None:
        self.ppl_model = aux_model
        self.sim_model = sim_model

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

        rag = DummyRag(self.ppl_model, a1)

        signals_yeses = self._get_signals(rag, a1)
        yeses = {
            "sims": [signals_yeses[k]["sim"] for k in signals_yeses],
            "ppls": [signals_yeses[k]["ppl"] for k in signals_yeses],
        }

        signals_noes = self._get_signals(rag, a4)
        noes = {
            "sims": [signals_noes[k]["sim"] for k in signals_noes],
            "ppls": [signals_noes[k]["ppl"] for k in signals_noes],
        }

        print("pick best threshold")
        max_score = -1.0
        # try each possible threshold
        threshes = [
            (a, b) for a in noes["sims"] + yeses["sims"] for b in noes["ppls"] + yeses["ppls"]
        ]
        for ta, tb in threshes:
            def good(a: float, b: float) -> bool:
                return a >= ta and b <= tb

            tp = sum([1 if good(a, b) else 0 for a, b in zip(yeses["sims"], yeses["ppls"])])
            fp = sum([1 if good(a, b) else 0 for a, b in zip(noes["sims"], noes["ppls"])])
            tn = sum([1 if not good(a, b) else 0 for a, b in zip(noes["sims"], noes["ppls"])])
            fn = sum([1 if not good(a, b) else 0 for a, b in zip(yeses["sims"], yeses["ppls"])])
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            score = tpr - fpr
            if score > max_score:
                max_score = score
                best_ta, best_tb = ta, tb
                print(
                    f"new best ta: {best_ta}, best tb: {best_tb} with score {max_score} and tpr: {tpr}, fpr: {fpr} so thresh {(tpr+fpr)/2}"
                )
        print(f"best ta: {best_ta}, best tb: {best_tb} with score {max_score}")
        print("Done training.")

    def register_corpus(self, corpus: List[Tuple[str, str]]) -> None:
        self.corpus = corpus  # list of (id, doc) tuples

    def get_corpus_for_rag(self) -> List[Tuple[str, str]]:
        return self.corpus

    # Public: assumes corpus is already registered AND method is trained
    def infer_membership(self, rag_system: RagSystem) -> Tuple[bool, float, Any]:
        signals = self._get_signals(rag_system, self.corpus)

        good, total = 0, 0
        bitstring = ""
        for idd, signal in signals.items():
            sim, ppl = signal["sim"], signal["ppl"]
            if sim >= self.sim_thresh and ppl <= self.ppl_thresh:
                good += 1
                bitstring += "1"
            else:
                bitstring += "0"
            total += 1
        score = good / total
        return score > self.threshold, score, bitstring

    def compute_similarity(self, ref: str, cand: str) -> float:
        ref_chunks = ref.strip().split("\n\n")
        cand_chunks = cand.strip().split("\n\n")
        ref_embs = self.sim_model.encode(ref_chunks)
        cand_embs = self.sim_model.encode(cand_chunks)

        avg_cossim = 0
        for i, ref_emb in enumerate(ref_embs):
            maxsim = -2
            best = -1
            for j, cand_emb in enumerate(cand_embs):
                sim = dot(ref_emb, cand_emb) / (norm(ref_emb) * norm(cand_emb))
                if sim > maxsim:
                    maxsim = sim 
                    best = j
            avg_cossim += maxsim
        avg_cossim /= len(ref_embs)  # type: ignore
        return avg_cossim

    def _get_signals(
        self, rag_system: RagSystem | DummyRag, targets: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        rag_queries = []
        explicit_ids = []
        for idd, doc in targets:
            rag_queries.append(
                f"Continue this, writing a longer response: {doc[:200]}"
            )  # context to seed (unspecified)
            explicit_ids.append(idd)
        # Query RAG and get the score
        if not self.give_explicit_ids:
            explicit_ids = None

        signals = {}
        rag_responses = rag_system.generate(rag_queries, explicit_ids=explicit_ids)
        it = 0
        for (idd, doc), q, resp in zip(targets, rag_queries, rag_responses):
            sim = self.compute_similarity(resp, doc)
            curr_ppls, _ = self.ppl_model.get_ppls_and_logitinfo([q], [resp], logit_processors=None)
            ppl = curr_ppls[0].item()
            signals[idd] = {"sim": sim, "ppl": ppl}
        return signals
