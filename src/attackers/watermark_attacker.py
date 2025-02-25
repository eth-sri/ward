import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from transformers import LogitsProcessor, LogitsProcessorList

from src.config import AttackerConfig, MetaConfig
from src.models import APIModel, HfModel
from src.rag_system import RagSystem
from src.watermarks import BaseWatermark, get_watermark


class WatermarkAttacker:

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

        self.data: Dict[Any] = []  # based on corpus
        self.give_explicit_ids = self.cfg.give_explicit_ids

    def set_model(self, model: HfModel) -> None:
        self.para_model = model
        self.watermark: BaseWatermark = get_watermark(
            self.meta_cfg, self.cfg.watermark, self.para_model.tokenizer, self.cfg.model.name
        )
        self.gpt = APIModel("gpt4o")

    def register_corpus(self, corpus: List[Tuple[str, str]]) -> None:
        cache_filename = f"cache/watermark_attacker_{self.cfg.watermark.generation.seeding_scheme}_{round(self.cfg.watermark.generation.delta, 1)}.json"
        if not os.path.exists(cache_filename): 
            with open(cache_filename, "w") as f:
                json.dump({}, f)

        with open(cache_filename, "r") as f:
            cache = json.load(f)

        paraphraser_prompt = f"You are an expert rewriter. Rewrite the following document keeping its meaning and fluency and especially length. It is crucial to retain all factual information in the original document. DO NOT MAKE THE TEXT SHORTER. Do not start your response by 'Sure' or anything similar, simply output the paraphrased document directly. Do not add stylistic elements or anything similar, try to be faithful to the original content and style of writing. Do not be too formal. Keep all the factual information."
        gpt_qgen_prompt = f"Given a document, generate a question that can only be answered by reading the document. The answer should be a longer detailed response, so avoid factual and simple yes/no questions and steer more towards questions that ask for opinions or explanations of events or topics described in the documents. Do not provide the answer, provide just the question."
        response_format = None
        if self.cfg.queries_per_doc > 1:
            # replace this
            gpt_qgen_prompt = f"Given a document, generate exactly {self.cfg.queries_per_doc} questions that can only be answered by reading the document. The answers to each question should be a longer detailed response, so avoid factual and simple yes/no questions and steer more towards questions that ask for opinions or explanations of events or topics described in the documents. Do not provide the answers, provide just the questions. Return the result as a JSON object that contains one list named 'questions' that contains exactly {self.cfg.queries_per_doc} questions."
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "questions": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["questions"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            }

        ### Cut if we have qpd>1
        if self.cfg.queries_per_doc > 1:
            docs_needed = len(corpus) // self.cfg.queries_per_doc
            assert (
                len(corpus) % self.cfg.queries_per_doc == 0
            ), "Number of documents must be divisible by queries_per_doc"
            corpus = corpus[:docs_needed]
            print(
                f"Due to {self.cfg.queries_per_doc} queries per doc, we will only use {docs_needed} documents."
            )

        # Paraphrase each doc in the corpus to register it
        # Also generate the question with gpt
        data = {}
        para_queries: List[Tuple[str, str]] = []
        for idd, doc in corpus:
            if idd not in cache or "docwm" not in cache[idd]:
                para_queries.append((idd, f"{paraphraser_prompt}\nDOCUMENT:\n{doc}"))
            else:
                data[idd] = {"docwm": cache[idd]["docwm"]}

        # Parse paras and save to cache
        batch_size = 8
        for i in range(0, len(para_queries), batch_size):
            batch = para_queries[i : min(len(para_queries), i + batch_size)]
            ids, docs = zip(*batch)
            docs_wm = self.para_model.generate(
                docs,  # type: ignore
                LogitsProcessorList(
                    [self.watermark.spawn_logits_processor(self.cfg.watermark.generation.delta)]  # type: ignore
                ),
            )[0]
            for idd, docwm in zip(ids, docs_wm):
                data[idd] = {"docwm": docwm}
                if idd not in cache:
                    cache[idd] = {}
                cache[idd]["docwm"] = docwm
            print(f"Done paraphrasing batch {i} to {i+batch_size} to insert the watermark")
        with open(cache_filename, "w") as f:
            json.dump(cache, f, indent=4)
        print("Saved docwm to cache")

        # Now generate questions
        qkey = (
            "question" if self.cfg.queries_per_doc == 1 else f"{self.cfg.queries_per_doc}questions"
        )
        gpt_queries: List[Tuple[str, str]] = []
        for idd, doc in corpus:
            if idd not in cache or qkey not in cache[idd]:
                gpt_queries.append((idd, f"DOCUMENT:\n{doc}"))
            else:
                data[idd][qkey] = cache[idd][qkey]

        gpt_responses = self.gpt.generate(
            gpt_qgen_prompt, [q[1] for q in gpt_queries], response_format=response_format
        )
        for (idd, _), resp in zip(gpt_queries, gpt_responses):
            if self.cfg.queries_per_doc > 1:
                resp = json.loads(resp)["questions"]
                # NOTE: sometimes there will not be 50 Qs (esp. for big values of 50) but just retry and nuke cache
            data[idd][qkey] = resp
            cache[idd][qkey] = resp

        with open(cache_filename, "w") as f:
            json.dump(cache, f, indent=4)
        print(f"Saved cache with questions")
        print("Done.")
        self.data = data  # dict of idd -> {"docwm": docwm, "question": question}

    def get_corpus_for_rag(self) -> List[Tuple[str, str]]:
        corpus = [(idd, self.data[idd]["docwm"]) for idd in self.data.keys()]
        if self.cfg.dropout_percent > 0:
            # shuffle the corpus collections
            random.shuffle(corpus)
            corpus = corpus[: round(len(corpus) * (1 - self.cfg.dropout_percent / 100.0))]
            print(f"Dropped out, so sending only {len(corpus)} docs to RAG")
        return corpus  # preprocesed

    def _filter_out_detector_dict(self, d: Dict) -> Dict:
        del d["offset_mapping"]
        del d["z_score_at_T"]
        del d["token_mask"]
        return d

    # no signals here, directly infer membership globally---accepts only preprocessed data above
    def infer_membership(self, rag_system: RagSystem) -> Tuple[bool, float, Any]:
        # Build RAG queries
        rag_queries = []
        explicit_ids = []

        if self.cfg.queries_per_doc == 1:
            for idd in self.data.keys():
                rag_queries.append(f"{self.data[idd]['question']}")
                explicit_ids.append(idd)
        else:
            qkey = f"{self.cfg.queries_per_doc}questions"
            for K in range(self.cfg.queries_per_doc):
                for idd in self.data.keys():
                    rag_queries.append(f"{self.data[idd][qkey][K]}")
                    explicit_ids.append(idd)
            # Do it like epochs -- all docs 1 -- all docs 2; for more informative curves

        assert len(rag_queries) == 200, f"Expected 200 queries, got {len(rag_queries)}"
        assert len(explicit_ids) == 200, f"Expected 200 ids, got {len(explicit_ids)}"

        # Query RAG and get the score
        if not self.give_explicit_ids:
            explicit_ids = None

        rag_responses = rag_system.generate(rag_queries, explicit_ids=explicit_ids)
        def get_detector_result(text: str) -> Dict[str, Any]:
            output_dict = self.watermark.detect([text])[0]
            output_dict = self._filter_out_detector_dict(output_dict)  # remove big fields
            return output_dict

        pvalues_each_20 = []
        toks_scored_each_20 = []
        for fst in range(20, len(rag_responses), 20):
            res = get_detector_result("\n".join(rag_responses[:fst]))
            pvalues_each_20.append(res["p_value"])
            toks_scored_each_20.append(res["num_tokens_scored"])

        resqqq = get_detector_result("\n".join(rag_queries))
        with open("results.txt", "a") as f:
            f.write(f"Piggyback with some rag queries detctor result: {resqqq}\n")

        res = get_detector_result("\n".join(rag_responses))

        # Added the full value now so don't need to do it in plots later
        pvalues_each_20.append(res["p_value"])
        toks_scored_each_20.append(res["num_tokens_scored"])

        # Add toks scored each 20 at the end of extra info
        extra_info = f"{str(pvalues_each_20)};{str(toks_scored_each_20)}"

        return res["z_score"] > 4.0, res["p_value"], extra_info
