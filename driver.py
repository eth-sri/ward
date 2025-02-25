import json
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.attackers import SibAttacker, get_attacker
from src.config import AttackerAlgo, RagWmConfig
from src.models import APIModel, HfModel, PspModel
from src.rag_system import RagSystem


def load_data(data_dir: str) -> List[Dict[str, Any]]:
    num_test_examples = 3391  # NOTE: 1000 for paper results
    # Load data
    data = []
    for i in range(num_test_examples):
        filename = f"{data_dir}/{i:04}.json"
        with open(filename, "r") as f:
            data.append(json.load(f))
    return data


def train_attackers(cfg: RagWmConfig, data_dir: str) -> None:
    # Load data, last 200 after 3391
    num_test_examples = 3391 # NOTE: 1000 for paper results
    num_train_examples = 200 
    data = []
    for idx in range(num_train_examples):
        i = idx + num_test_examples
        filename = f"{data_dir}/{i:04}.json"
        with open(filename, "r") as f:
            data.append(json.load(f))

    for algo in [AttackerAlgo.AAG, AttackerAlgo.SIB, AttackerAlgo.FACTS]:
        print(f"Training attacker {algo}")
        attacker = get_attacker(cfg, algo)
        attacker.train(data)
    print("DONE.")


def quality_check(cfg: RagWmConfig, data_dir: str, cache_filenames: Dict[float, str]) -> None:
    data = load_data(data_dir)
    docs_keyed = {}
    for i, d in enumerate(data):
        for model in ["gpt4o", "claude3.5sonnet", "llama3.1-405b", "qwen1.5-110b"]:
            docs_keyed[f'{d["id"]}_{model}'] = d["articles"][model]["article"]

    # Load caches
    caches = {}
    for delta, cache_file in cache_filenames.items():
        with open(cache_file, "r") as f:
            full_cache = json.load(f)
        # Get random 100 keys from cache
        keys = list(full_cache.keys())
        random.Random(123).shuffle(keys)
        keys = keys[:100]

        caches[delta] = {}
        for k in full_cache:
            if k in keys:
                caches[delta][k] = full_cache[k]["docwm"]
        assert len(caches[delta]) == 100

    gpt = APIModel("gpt4o")

    def parse_grade(q: str) -> int:
        try:
            q = q.split("GRADE:")[-1].strip()
            q = int(q)
            return q
        except:
            if len(q) <= 3 and q[-1] == ".":
                try:
                    return int(q[:-1])
                except:
                    return None  # could not parse really
            else:
                return None  # could not parse really

    def LOGqualityCheck(s: str) -> None:
        print(s)
        with open("qualitycheck.txt", "a") as f:
            f.write(s + "\n")

    ### (1) Grade independently and also PSP
    if False:
        pspmodel = PspModel()
        system_prompt = "Given a document, grade its quality and text fluency as an INTEGER on a scale of 1 to 10. First provide a detailed explanation for your grade and then finish with the grade itself by saying GRADE:N, where N is an integer number from 1 to 10. Do not finish the response with a period; finish exactly with the number."

        for delta, cache in caches.items():
            origs = []
            wms = []
            for k, v in cache.items():
                if k not in docs_keyed:
                    raise RuntimeError("wtf")
                orig = docs_keyed[k]
                wm = v
                origs.append(orig)
                wms.append(wm)
            N = len(origs)
            LOGqualityCheck(f"{delta}\nGrading {N} documents")

            origs_grades = gpt.generate(system_prompt, origs)
            wms_grades = gpt.generate(system_prompt, wms)
            psps = []
            for orig, wm in zip(origs, wms):
                psps.append(round(pspmodel.get_psp(orig, wm), 5))
            LOGqualityCheck(f"Mean psp: {sum(psps)/len(psps)}")

            orig_grades = [parse_grade(q) for q in origs_grades]
            wm_grades = [parse_grade(q) for q in wms_grades]
            # print mean
            LOGqualityCheck(f"Orig mean: {sum(orig_grades) / len(orig_grades)}")
            LOGqualityCheck(f"WM mean: {sum(wm_grades) / len(wm_grades)}")
            o_better, w_better, same = 0, 0, 0
            for og, wg in zip(orig_grades, wm_grades):
                if og > wg:
                    o_better += 1
                elif wg > og:
                    w_better += 1
                else:
                    same += 1
            LOGqualityCheck(f"Orig better: {o_better/N}, WM better: {w_better/N}, Same: {same/N}")
            LOGqualityCheck(f"PSPs: {psps}")
            LOGqualityCheck(f"Orig grades: {orig_grades}")
            LOGqualityCheck(f"WM grades: {wm_grades}")
            LOGqualityCheck("\n")

    ### (2) Same quality but of responses
    if True:
        LOGqualityCheck("Quality of responses?")
        intes_non_all = {}
        intes_yes_all = {}
        for tag in ["nowm", "wm"]:
            with open(f"quality_responses.txt", "r") as f:
                lines = f.readlines()

            it = 1
            intes_non = []
            intes_yes = []
            for intes in [intes_non, intes_yes]:
                for i in range(200):
                    curr = []
                    assert lines[it].strip().startswith("###SYS")
                    curr.append(lines[it])
                    it += 1
                    while not lines[it].strip().startswith("###SYS") and not lines[
                        it
                    ].strip().startswith("###RAG"):
                        curr.append(lines[it])
                        it += 1
                    intes.append("\n".join(curr))
                it += 1  # for rag
            intes_non_all[tag] = intes_non
            intes_yes_all[tag] = intes_yes

        intes_nowm = intes_yes_all["nowm"]
        intes_wm = intes_yes_all["wm"]

        system_prompt = "You are given an interaction of a user with a question answering system. ###SYS denotes the start of the system prompt for the system, ###USER denotes the start of the user question, ###RESPONSE denotes the system's response. Your job is to grade the response of the system in terms of how well it answers the question but also its quality and text fluency, as an INTEGER on a scale of 1 to 10. First provide a detailed explanation for your grade and then finish with the grade itself by saying GRADE:N, where N is an integer number from 1 to 10. Do not finish the response with a period; finish exactly with the number."

        gpt_output_nowm = gpt.generate(system_prompt, intes_nowm)
        gpt_output_wm = gpt.generate(system_prompt, intes_wm)

        gpt_responses_nowm = [q for q in gpt_output_nowm]
        gpt_responses_wm = [q for q in gpt_output_wm]
        grades_nowm = [parse_grade(q) for q in gpt_responses_nowm]
        grades_wm = [parse_grade(q) for q in gpt_responses_wm]

        LOGqualityCheck(f"Mean grade NOWM: {sum(grades_nowm)/len(grades_nowm)}")
        LOGqualityCheck(f"Mean grade WM: {sum(grades_wm)/len(grades_wm)}")
        LOGqualityCheck("NOWM")
        LOGqualityCheck(str(grades_nowm))
        LOGqualityCheck("WM")
        LOGqualityCheck(str(grades_wm))
        LOGqualityCheck("Detailed for NOWM")
        for x, y in zip(gpt_responses_nowm, grades_nowm):
            LOGqualityCheck(f"Grade {y}:\n{x}")

        LOGqualityCheck("Detailed for WM")
        for x, y in zip(gpt_responses_wm, grades_wm):
            LOGqualityCheck(f"Grade {y}:\n{x}")


def LOG(s: str) -> None:
    print(s)
    with open("results.txt", "a") as f:
        f.write(s + "\n")

def settingone(cfg: RagWmConfig, data_dir: str, rng: random.Random, aux_model: HfModel, sim_model: SentenceTransformer) -> None:
    data = load_data(data_dir)
    num_test_examples = 3391 # NOTE: 1000 for paper results

    # Reuse the rag model
    print("RAG model loading")
    try:
        rag_model = APIModel(cfg.rag.model.name)
    except:
        print("No API model, going to HF")
        rag_model = HfModel(cfg.meta, cfg.rag.model)  # type: ignore

    # Sample
    indices = list(range(num_test_examples))
    rng.shuffle(indices)
    print(f"[s1 checksum] {indices[:5]}")
    a1, a2, a3, a4 = [], [], [], []
    for i in range(200):
        a1.append(
            (f'{data[indices[i]]["id"]}_gpt4o', data[indices[i]]["articles"]["gpt4o"]["article"])
        )

    for i in range(200, 500):
        a2.append(
            (
                f'{data[indices[i]]["id"]}_claude3.5sonnet',
                data[indices[i]]["articles"]["claude3.5sonnet"]["article"],
            )
        )
    for i in range(500, 800):
        a3.append(
            (
                f'{data[indices[i]]["id"]}_llama3.1-405b',
                data[indices[i]]["articles"]["llama3.1-405b"]["article"],
            )
        )
    for i in range(800, 1000):
        a4.append(
            (
                f'{data[indices[i]]["id"]}_qwen1.5-110b',
                data[indices[i]]["articles"]["qwen1.5-110b"]["article"],
            )
        )

    for algo in cfg.attacker.algos:
        tag = f"### {cfg.rag.model.name},{cfg.rag.defended},setting1,{algo} ###"
        LOG(tag)

        attacker_nonmember = get_attacker(cfg, algo)
        if algo == AttackerAlgo.WATERMARK:
            attacker_nonmember.set_model(aux_model)
        if algo == AttackerAlgo.SIB:
            attacker_nonmember.set_models(aux_model, sim_model)
        attacker_nonmember.register_corpus(a4)

        attacker_member = get_attacker(cfg, algo)
        if algo == AttackerAlgo.WATERMARK:
            attacker_member.set_model(aux_model)
        if algo == AttackerAlgo.SIB:
            attacker_member.set_models(aux_model, sim_model)
        attacker_member.register_corpus(a1)

        ### Nonmember
        rag_corpus = a1 + a2 + a3
        rng.shuffle(rag_corpus)
        rag_system = RagSystem(cfg.meta, cfg.rag, rag_corpus, aux_model=rag_model)
        res, score, extra_info = attacker_nonmember.infer_membership(rag_system)
        LOG(f"Nonmember,{res},{score},{str(extra_info)}")

        ### Member
        rag_corpus = attacker_member.get_corpus_for_rag() + a2 + a3
        rng.shuffle(rag_corpus)
        rag_system = RagSystem(cfg.meta, cfg.rag, rag_corpus, aux_model=rag_model)
        res, score, extra_info = attacker_member.infer_membership(rag_system)
        LOG(f"Member,{res},{score},{str(extra_info)}")


def settingtwo(cfg: RagWmConfig, data_dir: str, rng: random.Random, aux_model: HfModel, sim_model: SentenceTransformer) -> None:
    data = load_data(data_dir)
    num_test_examples = 3391 # NOTE: 1000 for paper results

    indices = list(range(num_test_examples))
    rng.shuffle(indices)
    indices = indices[:1000]  # pick random 1000
    print(f"[s2 checksum] {indices[:5]}")

    a1, a4 = [], []
    rag_corpus_base = []
    a1_indices, a4_indices = list(indices), list(indices)
    rng.shuffle(a1_indices)
    rng.shuffle(a4_indices)
    N = 200
    a1_indices_set = set(a1_indices[:N])
    a4_indices_set = set(a4_indices[:N])
    for i in indices:
        for model in ["gpt4o", "claude3.5sonnet", "llama3.1-405b", "qwen1.5-110b"]:
            curr = (f'{data[i]["id"]}_{model}', data[i]["articles"][model]["article"])
            if model == "gpt4o":
                (
                    a1.append(curr) if i in a1_indices_set else rag_corpus_base.append(curr)
                )  # A1 comes later
            elif model == "qwen1.5-110b":
                if i in a4_indices_set:
                    a4.append(curr)
            else:
                rag_corpus_base.append(curr)  # only in RAG

    print("RAG model loading")
    try:
        rag_model = APIModel(cfg.rag.model.name)
    except:
        print("No API model, going to HF")
        rag_model = HfModel(cfg.meta, cfg.rag.model)  # type: ignore

    for algo in cfg.attacker.algos:
        tag = f"### {cfg.rag.model.name},{cfg.rag.defended},setting2,{algo} ###"
        LOG(tag)
        attacker_nonmember = get_attacker(cfg, algo)
        if algo == AttackerAlgo.WATERMARK:
            attacker_nonmember.set_model(aux_model)
        if algo == AttackerAlgo.SIB:
            attacker_nonmember.set_models(aux_model, sim_model)
        attacker_nonmember.register_corpus(a4)

        attacker_member = get_attacker(cfg, algo)
        if algo == AttackerAlgo.WATERMARK:
            attacker_member.set_model(aux_model)
        if algo == AttackerAlgo.SIB:
            attacker_member.set_models(aux_model, sim_model)
        attacker_member.register_corpus(a1)

        ### Nonmember
        rag_corpus = rag_corpus_base + a1 
        rng.shuffle(rag_corpus)
        rag_system = RagSystem(cfg.meta, cfg.rag, rag_corpus, aux_model=rag_model)
        res, score, extra_info = attacker_nonmember.infer_membership(rag_system)
        LOG(f"Nonmember,{res},{score},{str(extra_info)}")

        #### Member
        rag_corpus = rag_corpus_base + attacker_member.get_corpus_for_rag()  # A1 added
        rng.shuffle(rag_corpus)
        rag_system = RagSystem(cfg.meta, cfg.rag, rag_corpus, aux_model=rag_model)
        res, score, extra_info = attacker_member.infer_membership(rag_system)
        LOG(f"Member,{res},{score},{str(extra_info)}")
