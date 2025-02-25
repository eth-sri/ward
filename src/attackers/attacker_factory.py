from src.attackers.base_attacker import BaseAttacker
from src.attackers.facts_attacker import FactsAttacker
from src.attackers.aag_attacker import AagAttacker
from src.attackers.sib_attacker import SibAttacker
from src.attackers.watermark_attacker import WatermarkAttacker
from src.config import AttackerAlgo, RagWmConfig

algo_to_attacker = {
    AttackerAlgo.WATERMARK: WatermarkAttacker,
    AttackerAlgo.AAG: AagAttacker,
    AttackerAlgo.SIB: SibAttacker,
    AttackerAlgo.FACTS: FactsAttacker,
}


def get_attacker(cfg: RagWmConfig, algo: AttackerAlgo) -> BaseAttacker:
    return algo_to_attacker[algo](cfg.meta, cfg.attacker)
