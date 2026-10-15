# ICL corpus

Two kinds of file, both JSON objects keyed by the **verbatim training prompt**
(the same string the dataloader produces, so the keys must match the parquet
files in `$DATA_DIR` exactly).

### `*_similar_questions.json` — retrieval index

`prompt -> [nearest training prompts]`, ranked by Qwen3-Embedding-8B cosine
similarity (top 15, self excluded → 14 neighbours per prompt). Used to pick
which solved question to show as a demonstration.

### `*_icl_corpus_<model>.json` — demonstration pool

`prompt -> [known-correct solutions]`. Seeded from warm-up generations of the
policy model and **appended to during training**, so it is model-specific: a
Qwen3-1.7B run must not read a Qwen2.5 corpus.

## Files

| File | Keys | Non-empty | Values | Size |
| --- | --- | --- | --- | --- |
| `train_DAPO-Math-17k_similar_questions.json` | 17,396 | 17,396 | 243,544 | 97 MB |
| `train_deepcoder_similar_questions.json.zip` | 21,573 | 21,573 | — | 92 MB zipped, 804 MB unpacked |
| `train_DAPO-Math-17k_icl_corpus_qwen3_1.7b.json` | 17,396 | 13,289 | 23,656 | 62 MB |
| `train_deepcoder_icl_corpus_qwen3_1.7b.json` | 21,573 | 7,889 | 13,381 | 66 MB |
| `train_DAPO-Math-17k_icl_corpus_qwen2.5_7b.json` | 17,396 | 8,751 | 23,953 | 53 MB |
| `train_deepcoder_icl_corpus_qwen2.5_7b.json` | 21,573 | 6,155 | 17,191 | 80 MB |

The demonstration pools were seeded from 2 warm-up generations per prompt for
Qwen3-1.7B and 5 for Qwen2.5-7B-Instruct, over the full DAPO-Math-17k and
DeepCoder training sets. For any other policy model, regenerate the pool with
`nbs/generate_icl_corpus.ipynb`.

## Preparing them for a run

```bash
unzip train_deepcoder_similar_questions.json.zip -d .

python ../nbs/merge_icl_corpus.py \
    train_DAPO-Math-17k_similar_questions.json \
    train_deepcoder_similar_questions.json \
    -o train_merged_similar_questions.json

python ../nbs/merge_icl_corpus.py \
    train_DAPO-Math-17k_icl_corpus_qwen3_1.7b.json \
    train_deepcoder_icl_corpus_qwen3_1.7b.json \
    -o train_merged_icl_corpus_qwen3_1.7b.json
```

Point `rollout_policy.policy_kwargs.train_similar_questions` and
`.icl_corpus_path` at the two merged files. During training the policy writes a
snapshot to `$EXP_DIR/<experiment>/icl_corpus_<step>.json` every `save_freq`
steps.

## Regenerating from scratch

`nbs/generate_similar_questions.ipynb` (embeddings + top-k) and
`nbs/generate_icl_corpus.ipynb` (warm-up generations, scored with the same
reward functions used in training).

## Licensing

The keys are the verbatim question text of
[DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
and
[DeepCoder-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset);
the values in `*_icl_corpus_*` are Qwen model generations. See
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
