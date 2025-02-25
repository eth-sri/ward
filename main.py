import datetime
import gc
import json
import os
import random
import sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from driver import quality_check, settingone, settingtwo, train_attackers
from src.config import AttackerAlgo
from src.config.meta_config import get_pydantic_models_from_path
from src.models import APIModel, HfModel
from src.utils import create_open


def main(cfg_path: str) -> None:
    cfgs = get_pydantic_models_from_path(cfg_path)
    print(f"Number of configs: {len(cfgs)}")
    for cfg in cfgs:
        print(cfg)
        with open("results.txt", "a") as f: 
            f.write(f"{cfg}\n")

        out_dir = cfg.get_result_path()
        with create_open(f"{out_dir}/config.txt", "w") as f:
            json.dump(cfg.model_dump(mode="json"), indent=4, fp=f)

        # quality check
        if False:
            deltas = np.arange(0.5, 7.0, 0.5)
            caches = {}
            for delta in deltas:
                caches[delta] = (
                    f"plots/deltas/caches/watermark_attacker_ff-position_prf-2-False-1548585_{round(delta, 1)}.json"
                )
            quality_check(cfg, "farad", caches)
            exit()
        # train_attackers(cfg, "data")
        # exit()

        if AttackerAlgo.WATERMARK in cfg.attacker.algos or AttackerAlgo.SIB in cfg.attacker.algos:
            aux_model = HfModel(cfg.meta, cfg.attacker.model)
        else:
            aux_model = None

        if AttackerAlgo.SIB in cfg.attacker.algos:
            sim_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        else:
            sim_model = None

        data_dir = "farad"

        if cfg.eval.setting1:
            print("Setting 1 (Easy)")
            rng = random.Random(cfg.meta.seed)
            settingone(cfg, data_dir, rng, aux_model=aux_model, sim_model=sim_model)
        if cfg.eval.setting2:
            print("Setting 2 (Hard)")
            rng = random.Random(cfg.meta.seed)
            settingtwo(cfg, data_dir, rng, aux_model=aux_model, sim_model=sim_model)

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print(f"{datetime.datetime.now()}")
    if len(sys.argv) != 2:
        raise ValueError(
            f"Exactly one argument expected (the path to the config file), got {len(sys.argv)}."
        )
    main(sys.argv[1])
