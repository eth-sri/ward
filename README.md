# Ward 👨‍⚖️💧 <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>

This repository contains the code accompanying our ICLR 2025 [paper](https://www.sri.inf.ethz.ch/publications/jovanovic2025ward): 

> Nikola Jovanović, Robin Staab, Maximilian Baader and Martin Vechev. 2025. _Ward: Provable RAG Dataset Inference via LLM Watermarks._ In Proceedings of ICLR ’25.

## Installation

This project uses `uv` for package management. To install `uv`, run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

The dependencies will be installed on the first run of Ward.

### Model Names and API Keys
The models that are queried via API use the following keys that should be set as environment variables:
- OpenAI models (`gpt3.5`, `gpt4o`): `OPENAI_API_KEY`
- Anthropic models (`claude*`): `ANTHROPIC_API_KEY`
- Models hosted on Together (`llama*` and `qwen*`): `TOGETHER_API_KEY`

## Repository Structure 

The project structure is as follows.
- `main.py` is the main entry point for the code. The default code path runs the RAG-DI evaluation based on the provided config (calling `settingone()` and `settingtwo()` in `driver.py`). 
    - It can also be used to invoke baseline attacker training (`train_attackers(cfg, data_dir)` in `driver.py`), and the quality check experiment from Sec. 5.3 (`quality_check(cfg, data_dir, cache_filenames)` in `driver.py`).
- `datagen.py` was used to generate FARAD.
- `src/` contains the rest of the code, namely:
    - `src/attackers` contains implementations of Ward (`watermark_attacker.py`) and the baselines (FACTS, SIB, and AAG). 
    - `src/config` contains definitions of our Pydantic configuration files. Refer to `ragwm_config.py` for detailed explanations of each field.
    - `src/models` contains model classes for all our models (on both RAG provider and data owner side), and the P-SP paraphrasing quality evaluation.
    - `src/utils` contains utility functions for file handling and logging.
    - `src/watermarks` contains watermark implementations to be used by the data owner.
    - `rag_system.py` implements the RAG systems, both the simulated and the end-to-end one (Sec 5.3), including the MemFree decoding (Sec 5.2).
- `cache/` starts as empty files and will contain the cached intermediate steps of the methods. For more complex experiments make sure to clear this.
- `configs/` contains YAML configuration files (corresponding to `src/config/ragwm_config.py`) for our main experiments reported in the paper. 
    - `main_experiment.yaml` corresponds to our main results in Sec. 5.1 and 5.2.
    - `memfree.yaml` is used for our robustness experiment in Sec. 5.2.
    - `endtoend*.yaml` are used for our _Modeling retrieval_ experiments in Sec. 5.3.
    - `abl_wm_*.yaml` and `abl_rag_*.yaml` are used for the corresponding parts of our ablations in Sec. 5.4.
- `farad/` holds the FARAD dataset.

## Running the Code

Our code can be run with `uv` by providing a path to a YAML configuration file. For example,

```
uv run python3 main.py configs/example.yaml
```

will run the `FACTS` baseline with the Def-P prompt and `GPT3.5` as the RAG model, in the Hard setting. See `main_experiment.yaml` for a full configuration of our main experiments.

## Contact

Nikola Jovanović, nikola.jovanovic@inf.ethz.ch<br>
Robin Staab, robin.staab@inf.ethz.ch

## Citation

If you use our code please cite the following.

```
@inproceedings{jovanovic2025ward,
    author = {Jovanović, Nikola and Staab, Robin and Baader, Maximilian and Vechev, Martin},
    title = {Ward: Provable RAG Dataset Inference via LLM Watermarks},
    booktitle = {{ICLR}},
    year = {2025}
}
```
