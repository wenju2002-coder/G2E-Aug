# Retained Graph2Edits-Aug checkpoints

These are the original retained augmented-model checkpoints, not baseline
Graph2Edits weights. They are published directly in the directory structure used
by `eval_bucket.py`:

```text
uspto_50k/without_rxn_class/16-07-2026--01-17-31/epoch_86.pt
uspto_50k/with_rxn_class/15-07-2026--01-22-06/epoch_124.pt
```

The existing logs and bucket summaries are preserved. Both weights are
byte-identical to their original local files and to manuscript Additional file 6;
see the root `DATA_CHECKPOINTS_SHA256.json`. They were not retrained or modified
for this upload. No training-seed replication or new inference is claimed.

From the repository root, using the supplied matching processed data:

```bash
python eval_bucket.py --dataset uspto_50k --use_rxn_class False --experiments 16-07-2026--01-17-31 --epoch epoch_86.pt --expected_test_reactions 5004
python eval_bucket.py --dataset uspto_50k --use_rxn_class True --experiments 15-07-2026--01-22-06 --epoch epoch_124.pt --expected_test_reactions 5004
```

Historical checkpoint arguments and saved prior/co-occurrence buffers must be
preserved. The evaluator strictly loads them and never overwrites prior output
directories. `experiment/` (singular) is the retained-run directory; new training
runs use `experiments/` (plural).

Checkpoint formats can execute code when loaded. Only load trusted files after
verifying their provenance and hashes. Original Graph2Edits attribution and
licensing notices remain in LICENSE and the root README; this release does not
claim to relicense third-party model assets.
