# XRPO: Pushing the Limits of GRPO with Targeted Exploration and Exploitation

[![arXiv](https://img.shields.io/badge/arXiv-2510.06672-b31b1b.svg)](https://arxiv.org/abs/2510.06672)
[![ICML 2026](https://img.shields.io/badge/ICML-2026-blue.svg)](https://arxiv.org/abs/2510.06672)

Udbhav Bamba\*, Minghao Fang\*, Yifan Yu\*, Haizhong Zheng, Fan Lai
(\* equal contribution)

Official implementation of **XRPO** (e**X**plore–e**X**ploit GRPO), ICML 2026.
XRPO recasts GRPO policy optimization as a rollout exploration–exploitation
problem: it spends the rollout budget on the prompts where an extra sample is
most informative, seeds prompts that never receive reward with in-context
exemplars, and sharpens the advantage of correct but unlikely responses.

All three components live in
[`verl/trainer/ppo/rollout_policy/explore_exploit_rollout_policy.py`](verl/trainer/ppo/rollout_policy/explore_exploit_rollout_policy.py):

1. **Hierarchical rollout planning** (paper §3.2). Every prompt first gets
   `base_rollouts_per_prompt` rollouts. The remaining budget,
   `(rollout.n - base_rollouts_per_prompt) * train_batch_size`, is allocated
   over `dynamic_rounds` rounds by a priority score: the expected reduction of
   the Student-*t* confidence-interval half-width of the prompt's mean reward
   from one more rollout, plus an exploration bonus for sparsely sampled
   prompts.
2. **ICL seeding** (paper §3.2). In each allocation round, prompts that have no
   correct rollout yet spend their allocated rollouts on an in-context prompt:
   up to `icl_samples_per_prompt` solved exemplars of semantically similar
   training questions, taken from the ICL corpus, followed by the original
   question. Correct (non-ICL) rollouts are added back to the corpus, so it
   evolves with the policy.
3. **Novelty-guided advantage sharpening** (paper §3.3). For each correct
   response, novelty is `eta = exp(s(y) - mean_group s)`, with `s` the
   length-normalized log-likelihood. Its advantage becomes
   `A + clip(novelty_strength * (1 - eta), 0, novelty_clamp * A)`, so only
   correct responses that are less likely than their group average are boosted.

This repository is a fork of [verl](https://github.com/volcengine/verl)
`v0.4.0`. [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) lists exactly which
files differ from upstream.

## Install

```bash
git clone https://github.com/UIUC-MLSys/XRPO.git && cd XRPO
conda create -n xrpo python=3.10 -y && conda activate xrpo
pip install -e .
bash scripts/install_vllm_sglang_mcore.sh     # vLLM + megatron-core + math-verify
```

Training on code tasks executes model-generated programs. Install
[firejail](https://github.com/netblue30/firejail) first, and prefer an isolated
machine or container:

```bash
sudo apt-get install -y firejail
```

## Data

```bash
export DATA_DIR=./data
jupyter nbconvert --execute --inplace nbs/dapo_data.ipynb        # DAPO-Math-17k -> $DATA_DIR
jupyter nbconvert --execute --inplace nbs/deepcoder_data.ipynb   # DeepCoder     -> $DATA_DIR
```

## ICL corpus

XRPO needs two JSON files, both keyed by the *verbatim* training prompt:

| File | Content |
| --- | --- |
| `train_merged_similar_questions.json` | prompt -> its nearest training prompts (Qwen3-Embedding-8B) |
| `train_merged_icl_corpus_<model>.json` | prompt -> list of known-correct solutions, used as demonstrations |

Pre-generated files for the reported runs are in [`icl_corpus/`](icl_corpus).
`train_deepcoder_similar_questions.json` ships zipped (804 MB unpacked):

```bash
unzip icl_corpus/train_deepcoder_similar_questions.json.zip -d icl_corpus/
```

The policy takes a single file for the whole math+code mixture, so merge them:

```bash
python nbs/merge_icl_corpus.py \
    icl_corpus/train_DAPO-Math-17k_similar_questions.json \
    icl_corpus/train_deepcoder_similar_questions.json \
    -o icl_corpus/train_merged_similar_questions.json

python nbs/merge_icl_corpus.py \
    icl_corpus/train_DAPO-Math-17k_icl_corpus_qwen3_1.7b.json \
    icl_corpus/train_deepcoder_icl_corpus_qwen3_1.7b.json \
    -o icl_corpus/train_merged_icl_corpus_qwen3_1.7b.json
```

To regenerate the corpora from scratch:

```bash
jupyter nbconvert --execute --inplace nbs/generate_similar_questions.ipynb   # embeddings + top-k
jupyter nbconvert --execute --inplace nbs/generate_icl_corpus.ipynb          # warm-up generations, scored
```

## Train

```bash
cp launch_scripts/env.example.sh env.sh   # HF / wandb tokens, DATA_DIR, ICL_DIR, EXP_DIR
source env.sh

bash launch_scripts/explore_exploit_rollout_policy_qwen3_1.7b.sh    # XRPO, Qwen3-1.7B
bash launch_scripts/explore_exploit_rollout_policy_qwen2.5_14b.sh   # XRPO, Qwen2.5-14B-Instruct
bash launch_scripts/gspo_baseline_qwen3_1.7b.sh                     # GSPO baseline
```

On Slurm, adapt [`launch_scripts/slurm_template.sbatch`](launch_scripts/slurm_template.sbatch)
to your site and submit any of the scripts through it:

```bash
sbatch launch_scripts/slurm_template.sbatch launch_scripts/explore_exploit_rollout_policy_qwen3_1.7b.sh
```

### Configuration

XRPO is selected with `rollout_policy.policy_name=explore_exploit_rollout_policy`;
defaults live in [`verl/trainer/config/ppo_trainer.yaml`](verl/trainer/config/ppo_trainer.yaml).

| Key (under `rollout_policy.policy_kwargs`) | Default | Meaning |
| --- | --- | --- |
| `base_rollouts_per_prompt` | 4 | rollouts drawn for every prompt before allocation |
| `dynamic_rounds` | 2 | rounds the remaining budget is split over |
| `icl_enabled` | True | in-context exploration on/off |
| `icl_samples_per_prompt` | 2 | solved exemplars (K) placed in an ICL-seeded prompt |
| `train_similar_questions` | — | path to the nearest-neighbour map |
| `icl_corpus_path` | — | path to the demonstration corpus |
| `advantage_shaping_enabled` | True | novelty bonus on/off |
| `novelty_strength` | 2.5 | bonus scale (λ_novelty) |
| `novelty_clamp` | 0.5 | bonus ceiling, as a fraction of the response's own advantage (κ_clip) |

`policy_name=ppo_rollout_policy` gives plain GRPO/GSPO rollouts and is the
baseline configuration. `launch_scripts/gspo_baseline_qwen3_1.7b.sh` is the
GSPO baseline; for a GRPO baseline, override
`actor_rollout_ref.actor.policy_loss_type=vanilla` together with the clip
ratios of the XRPO script (`clip_ratio_low=0.2`, `clip_ratio_high=0.28`).

Two further keys are added on top of verl `v0.4.0`:
`actor_rollout_ref.actor.policy_loss_type` (`vanilla` or `gspo`) and
`data.enable_thinking` (passes `enable_thinking` through the chat template).

## Evaluate

```bash
MODEL=./experiments/<run>/global_step_<n>/actor bash evaluation/run_inference.sh
jupyter nbconvert --execute --inplace evaluation/calculate_scores_math.ipynb
jupyter nbconvert --execute --inplace evaluation/calculate_scores_code.ipynb
```

## Layout

```
verl/trainer/ppo/rollout_policy/    XRPO + the rollout policy interface
verl/utils/reward_score/            math (latex_math) and code (rllm_code) rewards
nbs/                                data preparation and ICL corpus generation
icl_corpus/                         pre-generated corpora for the reported runs
launch_scripts/                     training entry points
evaluation/                         inference + scoring
verl/                               the rest of the verl v0.4.0 fork
```

## Citation

```bibtex
@inproceedings{bamba2026xrpo,
  title     = {{XRPO}: Pushing the Limits of {GRPO} with Targeted Exploration and Exploitation},
  author    = {Bamba, Udbhav and Fang, Minghao and Yu, Yifan and Zheng, Haizhong and Lai, Fan},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026}
}
```

Please also cite verl:

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  journal = {arXiv preprint arXiv:2409.19256},
  year    = {2024}
}
```

## License

Apache 2.0 — see [`LICENSE`](LICENSE), [`Notice.txt`](Notice.txt) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
