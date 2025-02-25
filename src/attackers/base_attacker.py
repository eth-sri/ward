from typing import Any, Dict, List, Optional, Tuple

from src.attackers.dummy_rag import DummyRag
from src.config import AttackerConfig, MetaConfig
from src.rag_system import RagSystem


class BaseAttacker:

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
        self.cfg = attacker_cfg
        self.corpus = None
        pass

    def train(self, trainset: List[Dict[str, Any]]) -> Tuple[float, float]:
        raise NotImplementedError("BaseAttacker.train not implemented")

    def register_corpus(self, corpus: List[Tuple[str, str]]) -> None:
        raise NotImplementedError("BaseAttacker.register_corpus not implemented")

    def get_corpus_for_rag(self) -> List[Tuple[str, str]]:
        raise NotImplementedError("BaseAttacker.get_corpus_for_rag not implemented")

    def infer_membership(self, rag_system: RagSystem) -> Tuple[bool, float, Any]:
        raise NotImplementedError("BaseAttacker.infer_membership not implemented")

    # Internal abstraction to get signal for a corpus; so train and infer can be unified
    def _get_signals(
        self, rag_system: RagSystem | DummyRag, curr_corpus: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        raise NotImplementedError("BaseAttacker._get_signal not implemented")
