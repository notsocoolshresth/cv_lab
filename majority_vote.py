"""
Majority Voting Ensemble — combine 3 submission CSVs via majority vote.

Usage:
    python majority_vote.py csv1.csv csv2.csv csv3.csv [-o output.csv]

Each CSV must have columns: Id, Category
Output: a single CSV with the majority-voted Category per Id.
"""

import sys
import csv
import argparse
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Majority vote over 3 submission CSVs")
    parser.add_argument("csvs", nargs=3, help="Paths to 3 submission CSVs")
    parser.add_argument("-o", "--output", default="majority_vote_submission.csv",
                        help="Output CSV path (default: majority_vote_submission.csv)")
    args = parser.parse_args()

    # Read all 3 CSVs
    predictions = []
    for path in args.csvs:
        preds = {}
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                preds[row["Id"]] = row["Category"]
        predictions.append(preds)
        print(f"  Loaded {path}: {len(preds)} samples")

    # Verify all CSVs have the same Ids
    ids_0 = set(predictions[0].keys())
    for i, preds in enumerate(predictions[1:], 1):
        if set(preds.keys()) != ids_0:
            diff = ids_0.symmetric_difference(set(preds.keys()))
            print(f"  WARNING: CSV {i+1} has different Ids! Difference: {len(diff)} samples")

    # Majority vote
    all_ids = list(predictions[0].keys())  # preserve order from first CSV
    final = {}
    agree_count = 0
    disagree_ids = []

    for sample_id in all_ids:
        votes = [preds[sample_id] for preds in predictions if sample_id in preds]
        counter = Counter(votes)
        winner, count = counter.most_common(1)[0]
        final[sample_id] = winner

        if count == 3:
            agree_count += 1
        else:
            disagree_ids.append((sample_id, votes, winner))

    # Write output
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Category"])
        for sample_id in all_ids:
            w.writerow([sample_id, final[sample_id]])

    # Summary
    print(f"\n  Total samples: {len(all_ids)}")
    print(f"  All 3 agree:   {agree_count} ({100*agree_count/len(all_ids):.1f}%)")
    print(f"  Disagreements: {len(disagree_ids)} ({100*len(disagree_ids)/len(all_ids):.1f}%)")

    if disagree_ids:
        # Show class distribution of disagreements
        print(f"\n  Disagreement details (first 20):")
        for sid, votes, winner in disagree_ids[:20]:
            print(f"    {sid}: {votes} → {winner}")

    # Show final distribution
    final_dist = Counter(final.values())
    print(f"\n  Final distribution: {dict(final_dist)}")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
