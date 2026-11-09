# G2E-Aug

Complete source for **Graph2Edits-Aug**, an augmented Graph2Edits implementation
used in **RetroTail: Edit-frequency stratified evaluation for single-step
retrosynthesis prediction**.

The previous review-period placeholders have been replaced with the complete
model, training, preprocessing and chemistry implementations. This release
restores the manuscript's archived implementation, without new training results
or changes to the model architecture.

RetroTail is an **evaluation workflow**, not a new predictor. The case study
covers the Graph2Edits family, one USPTO-50K split and one checkpoint per model
and class setting. Low reference-edit frequency is not chemical novelty, and
cross-model generality has not been established.

## Upstream project

Derived from [Graph2Edits](https://github.com/Jamson-Zhong/Graph2Edits):

> Zhong W, Yang Z, Chen CY-C. Retrosynthesis prediction using an end-to-end graph
> generative architecture for molecular graph editing. Nature Communications
> 14, 3009 (2023). https://doi.org/10.1038/s41467-023-38851-5

The upstream release is on [Zenodo](https://doi.org/10.5281/zenodo.7837349).
That DOI belongs to original Graph2Edits, not this project. Please cite and
retain the upstream work and its license.

## Environment requirements

| Package | Reference version |
| --- | --- |
| Python | 3.11.8 |
| PyTorch | 2.2.2 |
| NumPy | 1.26.4 |
| RDKit | 2024.03.4 |

Additional dependencies: pandas, joblib and tqdm. From the repository root:

```bash
conda create -n g2e-aug python=3.11.8
conda activate g2e-aug
pip install -r requirements.txt
```

These are author-supplied reference versions, not a fully locked historical
environment. Select a PyTorch CPU/CUDA build compatible with your machine.
Software versions, particularly RDKit canonicalization, may affect inference.
Fresh-environment reproduction of all historical runs is not claimed here.

## Repository layout

```text
models/                         complete augmented model, encoder and beam search
utils/                          chemistry, graph, frequency and balancing helpers
data/Atoms_character.txt        electronegativity lookup
data/Bond_Energy.txt             bond-energy lookup
data/uspto_50k/                 raw, canonicalised and processed matching splits
canonicalize_prod.py             canonicalize and remap raw reaction CSV files
preprocess.py                   generate reference edits and training artifacts
prepare_data.py                 tensorize training data
train.py                        train Graph2Edits-Aug
eval.py                         retained legacy aggregate evaluator
eval_bucket.py                  recommended explicit-checkpoint evaluator
historical_eval_bucket.py       retained historical bucket evaluator
export_reaction_predictions.py  optional standalone prediction exporter
cils_statistical_resolution.py  legacy unpaired power-planning calculation
reaction_class_mapping.csv      index 0-9 to standard reaction class 1-10
experiment/                     two retained Aug checkpoints, logs and reports
DATA_CHECKPOINTS_SHA256.json     hashes and source provenance for published assets
tests/                          lightweight evaluator regression tests
```

## Data and checkpoints

The original directory layout now directly includes the **Graph2Edits-Aug
datasets and its two retained checkpoints**. No ZIP extraction or extra
`reproducibility/` directory is required. The version is tagged
[`g2e-aug-data-20261109`](https://github.com/wenju2002-coder/G2E-Aug/tree/g2e-aug-data-20261109).

| Location | Contents |
| --- | --- |
| [data/uspto_50k/](data/uspto_50k/) | Raw and canonicalised train/validation/test CSVs |
| [data/uspto_50k/train/](data/uspto_50k/train/) | Processed training reactions, edit/atom/leaving-group vocabularies and frequency map |
| [data/uspto_50k/valid/](data/uspto_50k/valid/) | Matching processed validation reactions |
| [data/uspto_50k/test/](data/uspto_50k/test/) | Matching processed test reactions |
| [Unknown-class checkpoint](experiment/uspto_50k/without_rxn_class/16-07-2026--01-17-31/epoch_86.pt) | Aug epoch 86, without reaction-class input |
| [Known-class checkpoint](experiment/uspto_50k/with_rxn_class/15-07-2026--01-22-06/epoch_124.pt) | Aug epoch 124, with reaction-class input |

The six CSVs and both checkpoints are byte-identical to the original local
Graph2Edits-Aug resources. The processed files and vocabularies match manuscript
Additional file 5; the checkpoints also match Additional file 6. File hashes and
provenance are in [DATA_CHECKPOINTS_SHA256.json](DATA_CHECKPOINTS_SHA256.json).
No new training or inference was performed for this upload. The retained
Graph2Edits **baseline** checkpoints and the four model-setting prediction tables
remain in the manuscript's Additional files; they are not Aug checkpoints.
Local environments and manuscript files are not included. No project-specific
Zenodo DOI is claimed.

Clone the repository, install the reference dependencies and run either explicit
evaluation command below using the supplied files. No preprocessing is needed
to use these retained processed evaluation inputs. Do not overwrite them with a
new preprocessing run if you intend to audit this exact archived split. Only
load trusted checkpoint/joblib files: their formats can execute code.

If intentionally rebuilding from the supplied raw CSVs, the required columns
are `id`, `class` and `reactants>reagents>production`. Canonicalisation commands:

```bash
python canonicalize_prod.py --dataset uspto_50k --mode train
python canonicalize_prod.py --dataset uspto_50k --mode valid
python canonicalize_prod.py --dataset uspto_50k --mode test
```

Use matching processed files for exact reproduction; different chemistry
versions need not generate identical records. The audited split has 39985
processed training reactions and 5004 evaluated test reactions; three source
test reactions with unsupported `Add Bond` edits are filtered in preprocessing.

## Preprocessing and training

Run from the repository root, training split first:

```bash
python preprocess.py --mode train --dataset uspto_50k
python preprocess.py --mode valid --dataset uspto_50k
python preprocess.py --mode test --dataset uspto_50k
```

Training preprocessing creates vocabularies, `action_freq_map.pkl`,
`lg_cooccurrence_adj.pt` and `edit_class_stats.pt` in `data/uspto_50k/train/`.

Reaction class unknown (class input withheld):

```bash
python prepare_data.py --dataset uspto_50k
python train.py --dataset uspto_50k
```

Reaction class known (class input supplied):

```bash
python prepare_data.py --dataset uspto_50k --use_rxn_class
python train.py --dataset uspto_50k --use_rxn_class
```

In preparation/training, `--use_rxn_class` is a store-true flag: include it for
known and omit it for unknown; do not append `False`.
Training writes to `experiments/` (plural), whereas archived run folders use
`experiment/` (singular). The recommended evaluator accepts an explicit
checkpoint path, so no directory renaming is needed.

## Evaluation

### Recommended interface: eval_bucket.py

This evaluator strictly loads an explicit checkpoint, preserves its saved
prior/co-occurrence buffers and reports aggregate and stratified Top-1/3/5/10
accuracy. It also exports per-reaction ranked records without changing beam
ranking or training. After obtaining the matching weights/data:

```bash
# Reaction class unknown
python eval_bucket.py --dataset uspto_50k --use_rxn_class False --experiments 16-07-2026--01-17-31 --epoch epoch_86.pt --expected_test_reactions 5004

# Reaction class known
python eval_bucket.py --dataset uspto_50k --use_rxn_class True --experiments 15-07-2026--01-22-06 --epoch epoch_124.pt --expected_test_reactions 5004
```

Here `--experiments` resolves under `experiment/uspto_50k/` (singular). For a
new training run or another location, use `--checkpoint` **instead of**
`--experiments`/`--epoch`:

```bash
python eval_bucket.py --dataset uspto_50k --checkpoint experiments/uspto_50k/without_rxn_class/YOUR_RUN/epoch_YOUR_EPOCH.pt --output_dir evaluation/YOUR_NEW_RUN
```

Add `--use_rxn_class` for a class-conditioned checkpoint. This modern evaluator
accepts the flag alone or explicit `True`/`False` values. Existing output
directories are never overwritten. See `python eval_bucket.py --help` for
CPU/CUDA, beam, input-file and smoke-test options. A `--max_reactions` run is
explicitly partial, not a full benchmark.

Each new evaluation directory contains:

- `pred_results.txt`: human-readable ranked predictions;
- `predictions_top10.csv` and `predictions.jsonl`: per-reaction candidates,
  identifiers, bucket assignments and correctness flags;
- `bucket_eval_<epoch>.csv` / `.md`: aggregate and subgroup summaries;
- `metadata.json`: source/data/checkpoint hashes, versions and completion state;
- `action_freq_map.pkl`: the frequency map used for that evaluation.

Invalid candidate ranks and reaction denominators are retained. Bucket summaries
do not replace per-reaction records for paired model comparisons.

### Legacy interface: eval.py

The historical scoring implementation is retained unchanged. It writes
`pred_results.txt` or numbered `pred_results_*.txt` files, but uses the
hard-coded `epoch_86.pt` under `experiments/`. Its `--use_rxn_class` is a
store-true flag, not a Boolean-value argument. For other epochs, including
class-known epoch 124, use `eval_bucket.py` above; do not mistake the legacy
script for a generic checkpoint-selection interface.

## Frequency-stratified protocol

Each test reaction takes the minimum training count of its required
non-terminal reference edits; unseen labels have count zero:

- **Many-shot:** count >100;
- **Medium-shot:** count 20-100 inclusive;
- **Few-shot:** count <20.

The manuscript split has 4832 / 120 / 52 reactions in these groups. Grouping
requires the reference reactants, so it is retrospective, not a product-only
novelty detector. Processed class indices 0-9 map to standard classes 1-10.
Class known/unknown describes the model input, not missing annotations.

## Model scope

Seven jointly enabled changes are an atom-environment-electronegativity-derived
bond descriptor, a bond-energy scalar inside an auxiliary bond-breaking head,
cost-sensitive auxiliary loss, graph alignment, effective-number balancing,
prior-based logit adjustment during **training and inference**, and a
leaving-group co-occurrence graph. Bond energy is not appended to the main bond
encoder input and does not directly rank beam candidates. No single-component
ablation or seed-replicated improvement claim is made.

## Tests

```bash
python -m unittest discover -s tests -v
```

These check parsing, bucket boundaries, class numbering and rank-preserving
scoring without running model training or inference.

## License

The upstream [MIT License](LICENSE) and copyright notice are retained. Original
datasets, checkpoints and third-party materials retain their own terms; this
code license does not relicense them.
