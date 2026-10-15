"""Merge the per-dataset ICL corpus / similar-question files into one file.

The rollout policy takes a single JSON mapping for the whole training mixture,
so the math and code files produced by `generate_icl_corpus.ipynb` and
`generate_similar_questions.ipynb` have to be merged first.

    python nbs/merge_icl_corpus.py \
        icl_corpus/train_DAPO-Math-17k_similar_questions.json \
        icl_corpus/train_deepcoder_similar_questions.json \
        -o icl_corpus/train_merged_similar_questions.json
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="JSON files to merge (later files win on key collisions)")
    parser.add_argument("-o", "--output", required=True, help="Path of the merged JSON file")
    args = parser.parse_args()

    merged = {}
    for path in args.inputs:
        with open(path) as f:
            data = json.load(f)
        overlap = merged.keys() & data.keys()
        if overlap:
            print(f"{path}: {len(overlap)} keys already present, overwriting")
        merged.update(data)
        print(f"{path}: {len(data)} entries -> {len(merged)} total")

    with open(args.output, "w") as f:
        json.dump(merged, f)
    print(f"wrote {len(merged)} entries to {args.output}")


if __name__ == "__main__":
    main()
