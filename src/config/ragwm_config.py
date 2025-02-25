from enum import Enum
from typing import Any, List, Optional, Tuple, Union

from pydantic import Field, validator

from src.config.utils import PydanticBaseModelWithOptionalDefaultsPath as PBMwODP

"""
    Pre-detection normalizers from watermarks/kgw/normalizers.py
"""


class WatermarkDetectionNormalizer(Enum):
    UNICODE = "unicode"
    HOMOGLYPHS = "homoglyphs"
    TRUECASE = "truecase"


class WatermarkScheme(Enum):
    KGW = "kgw"


class MetaConfig(PBMwODP, extra="forbid"):  # type: ignore
    device: str = Field(..., description="Device to run on (cuda/cpu)")
    seed: int = Field(
        ..., description="Seed, applied before every unwatermarked/watermark generate call"
    )
    out_root_dir: str = Field(..., description="Directory to save outputs to")
    result_dir: str = Field(
        "results/default", description="Directory to save evaluation results to"
    )

    def short_str(self) -> str:
        return f"seed={self.seed}"


class ModelConfig(PBMwODP, arbitrary_types_allowed=True):  # type: ignore
    skip: bool = Field(..., description="If this model should be loaded or skipped for speed")
    name: str = Field(description="Name of the model (from hf or openai-)")
    use_fp16: bool = Field(False, description="Load and use in FP16 precision")
    use_flashattn2: bool = Field(False, description="Use flash attention")
    prompt_max_len: int = Field(None, description="Max length of the prompt")
    response_max_len: int = Field(None, description="Max length of the response")
    n_beams: int = Field(1, description="Number of beams for beam search")
    use_sampling: bool = Field(False, description="Use multinomial sampling instead of greedy")
    sampling_temp: float = Field(0.7, description="Temperature for multinomial sampling")

    def short_str(self) -> str:
        return f"name={self.name},n_beams={self.n_beams},sample={self.use_sampling},temp={self.sampling_temp}"


class WatermarkGenerationConfig(PBMwODP, extra="forbid"):  # type: ignore
    seeding_scheme: str = Field(..., description="Seeding scheme, see prf_schemes.py")
    gamma: float = Field(..., description="Fraction of green tokens")
    delta: float = Field(..., description="Logit boost for green tokens")


class WatermarkDetectionConfig(PBMwODP, extra="forbid"):  # type: ignore
    normalizers: List[WatermarkDetectionNormalizer] = Field(
        ..., description="Preprocessors/normalizers to apply"
    )
    ignore_repeated_ngrams: bool = Field(
        ..., description="If repetitions should be ignored when counting hits"
    )
    z_threshold: float = Field(..., description="Min z-score to consider a text watermarked")


class WatermarkConfig(PBMwODP, extra="forbid"):  # type: ignore
    scheme: WatermarkScheme = Field(..., description="Watermark scheme to use")
    generation: WatermarkGenerationConfig
    detection: WatermarkDetectionConfig

class AttackerAlgo(Enum):
    WATERMARK = "watermark"
    AAG = "aag"
    SIB = "sib"
    FACTS = "facts"


class AttackerConfig(PBMwODP, extra="forbid"):  # type: ignore
    algos: List[AttackerAlgo] = Field(..., description="Attacker algorithms to use")
    model: ModelConfig = Field(..., description="Model to use for the attacker")
    watermark: WatermarkConfig = Field(None, description="Watermark to use if wm attacker")
    give_explicit_ids: bool = Field(
        False, description="Should the attacker give IDs to cheat retrieval -- always"
    )
    queries_per_doc: int = Field(1, description="WATERMARK: How many queries per doc to use")
    dropout_percent: int = Field(
        0,
        description="WATERMARK: What pc of docs gets removed before I give my corpus to the RAG",
    )

    def short_str(self) -> str:
        return f"model=[{self.model.short_str()}],wm=[{self.watermark.scheme.value}]"


class RagConfig(PBMwODP, extra="forbid"):  # type: ignore
    model: ModelConfig = Field(..., description="Model to use for the server")
    embedder_name: str = Field(..., description="Embedder name")
    real_retrieval: bool = Field(False, description="If real retrieval should be used")
    n_shots: int = Field(False, description="How many shots do we do")
    defended: bool = Field(False, description="If the model should be defended")
    memfree_n: int = Field(-1, description="Memfree n-gram size if >0")


class EvalConfig(PBMwODP, extra="forbid"):  # type: ignore
    setting1: bool = Field(True, description="If setting 1 (easy) should be evaluated")
    setting2: bool = Field(True, description="If setting 2 (hard) should be evaluated")

    def short_str(self) -> str:
        return "eval"


class RagWmConfig(PBMwODP):
    meta: MetaConfig
    rag: RagConfig
    attacker: AttackerConfig
    eval: EvalConfig
    def get_result_path(self) -> str:
        return f"{self.meta.result_dir}/meta=[{self.meta.short_str()}]"
