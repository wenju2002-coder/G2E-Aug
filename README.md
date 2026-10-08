# G2E-Aug

Frequency-stratified evaluation for graph-editing single-step retrosynthesis.

This repository contains the complete G2E-Aug source implementation used for
training, preprocessing, inference, and frequency-stratified evaluation. The
data and checkpoint files needed to reproduce the archived manuscript results
are distributed separately because they are research artifacts rather than
source code.

## Overview

G2E-Aug is based on Graph2Edits and is used to study how average Top-k accuracy
can conceal performance differences between frequent and rare graph edits. The
repository supports two complementary evaluation views:

1. `eval.py` performs the conventional, unstratified beam-search evaluation and
   records reaction-level predictions.
2. `eval_bucket.py` performs an independent beam-search evaluation and reports
   Top-1, Top-3, Top-5, and Top-10 exact-match accuracy by edit-frequency bucket.

The two scripts do not consume each other's output. Run both when both the
conventional prediction record and the frequency-stratified report are needed.

The main contribution of the accompanying study is the evaluation protocol and
benchmark diagnosis. G2E-Aug is used as a case study; the available results do
not establish that any individual augmentation causes a tail-performance gain.

## Release status

The complete source implementation is included. This release contains the
augmented model, training and preprocessing pipelines, class-balancing and
chemistry utilities, conventional evaluation, and frequency-stratified
evaluation. Reproducing the archived numerical results additionally requires
the processed USPTO-50K artifacts and the corresponding model checkpoint; the
expected paths and file formats are documented below.

## Method summary

Relative to the Graph2Edits backbone, the research branch jointly enables:

1. a tabulated bond-dissociation-energy feature for an auxiliary reaction-centre
   head;
2. a cost-sensitive auxiliary bond-breaking loss;
3. product-reactant graph-representation alignment;
4. effective-number weighting for edit and leaving-group targets;
5. posterior logit adjustment for edit and leaving-group scores;
6. leaving-group co-occurrence initialization and neighbour regularization.

These changes were enabled together, without a component-wise ablation. The
energy-aware head contributes through training losses; beam search is still
ranked by the main edit logits.

## Repository layout

```text
data/
  Atoms_character.txt        atom-property lookup table
  Bond_Energy.txt            tabulated bond-energy lookup table
models/
  beam_search.py             beam-search inference
  encoder.py                 graph encoder components
  graph2edits.py             complete augmented model implementation
utils/
  action_frequencies.py      training-edit frequency counter
  collate_fn.py              graph batching for inference
  reaction_actions.py        graph edit actions
  rxn_graphs.py              molecular graph construction
  class_balance.py           long-tail balancing and logit adjustment
  chem.py                    chemistry and bond-energy feature helpers
preprocess.py                extract edit sequences and build artifacts
prepare_data.py              tensorize edit sequences for training
train.py                     train G2E-Aug
eval.py                      conventional overall evaluation
eval_bucket.py               frequency-stratified evaluation
experiment/                  archived logs and aggregate reports
```

The executable scripts read checkpoints from `experiments/` (plural), whereas
the report artifacts in this source release are stored in `experiment/`
(singular). Place downloaded checkpoints under the corresponding plural path
before evaluation, or update the selected experiment path consistently.

## Environment requirements

The validated core environment is:

| Package | Version |
| --- | ---: |
| Python | 3.11.8 |
| PyTorch | 2.2.2 |
| NumPy | 1.26.4 |
| RDKit | 2024.03.4 |

The code also imports `pandas`, `joblib`, and `tqdm`. Their versions are not
pinned by this release. A reference installation is:

```bash
conda create -n g2e-aug python=3.11.8
conda activate g2e-aug
conda install -c conda-forge rdkit=2024.03.4 numpy=1.26.4
python -m pip install torch==2.2.2 pandas joblib tqdm
```

Verify the core versions with:

```bash
python -c "import torch, numpy, rdkit; from rdkit import rdBase; print(torch.__version__, numpy.__version__, rdBase.rdkitVersion)"
```

CUDA is optional. Both evaluators select CUDA automatically when
`torch.cuda.is_available()` is true and otherwise run on CPU.

OpenNMT is not required for the reported exact-match evaluation. It is only
needed by the experimental `--round_trip` path in `eval_bucket.py`.

## Required evaluation artifacts

Run commands from the repository root. For `uspto_50k`, the evaluators expect a
layout equivalent to:

```text
data/uspto_50k/
  test/test.file.kekulized
  train/action_freq_map.pkl
  train/lg_cooccurrence_adj.pt
  train/edit_class_stats.pt
experiments/uspto_50k/
  without_rxn_class/<run-name>/<checkpoint>.pt
  with_rxn_class/<run-name>/<checkpoint>.pt
```

Each checkpoint must contain the model configuration under `saveables` and the
parameter state under `state`. The processed test file is loaded with `joblib`.
Predicted and reference reactants are compared as canonicalized component sets
after atom-mapping information is removed.

If `action_freq_map.pkl` is absent, `eval_bucket.py` calls
`utils/action_frequencies.py` to rebuild it from the processed training data.
That fallback requires `data/uspto_50k/train/train.file.kekulized`.

## Evaluation 1: conventional results with `eval.py`

`eval.py` performs the upstream-style evaluation over the complete test set. It
runs beam search, reports running Top-1, Top-3, Top-5, and Top-10 accuracy, and
writes every candidate prediction with its probability and edit sequence.

Reaction class unknown:

```bash
python eval.py \
  --dataset uspto_50k \
  --experiments 16-07-2026--01-17-31 \
  --epoch epoch_86.pt \
  --beam_size 10 \
  --max_steps 9
```

Reaction class known:

```bash
python eval.py \
  --dataset uspto_50k \
  --use_rxn_class \
  --experiments 15-07-2026--01-22-06 \
  --epoch epoch_124.pt \
  --beam_size 10 \
  --max_steps 9
```

Use `--epoch` to select the checkpoint file in the chosen experiment directory.
Use the same checkpoint for conventional and frequency-stratified evaluation
when comparing their aggregate results.

The prediction record is saved as:

```text
experiments/<dataset>/<setting>/<run-name>/pred_results.txt
```

If the file already exists, `eval.py` creates `pred_results_1.txt`,
`pred_results_2.txt`, and so on instead of overwriting it.

## Evaluation 2: frequency-stratified results with `eval_bucket.py`

`eval_bucket.py` evaluates each reaction according to the least frequent
non-terminal ground-truth edit that it requires.

### Bucket definition

1. Count ground-truth edit actions in the training set.
2. Exclude the terminal `Terminate` action.
3. For each test reaction, take the minimum training frequency among its
   required edits.
4. Assign the reaction to one of three buckets:

   - **Many-shot:** minimum frequency greater than 100.
   - **Medium-shot:** minimum frequency from 20 through 100, inclusive.
   - **Few-shot:** minimum frequency below 20; unseen edits have frequency 0.

The evaluation unit is a reaction, not an individual edit action. The script
reports each bucket overall and within each USPTO reaction class.

Reaction class unknown:

```bash
python eval_bucket.py \
  --dataset uspto_50k \
  --experiments 16-07-2026--01-17-31 \
  --epoch epoch_86.pt \
  --beam_size 10 \
  --max_steps 9
```

Reaction class known:

```bash
python eval_bucket.py \
  --dataset uspto_50k \
  --use_rxn_class \
  --experiments 15-07-2026--01-22-06 \
  --epoch epoch_124.pt \
  --beam_size 10 \
  --max_steps 9
```

Optional output names can be supplied with:

```bash
python eval_bucket.py \
  --dataset uspto_50k \
  --experiments <run-name> \
  --epoch <checkpoint>.pt \
  --report_file bucket_report.md \
  --csv_file bucket_report.csv
```

By default, the script writes:

```text
bucket_eval_epoch_<N>.md
bucket_eval_epoch_<N>.csv
```

Existing reports are not overwritten; a numbered sibling is created. The CSV
contains evaluation metadata followed by:

```text
section,reaction_class,bucket,total_reactions,top1,top3,top5,top10,round_trip
```

## Which evaluator should be used?

| Goal | Script | Main output |
| --- | --- | --- |
| Inspect individual beam predictions | `eval.py` | `pred_results*.txt` |
| Report conventional aggregate Top-k accuracy | `eval.py` | console progress and prediction record |
| Measure many-, medium-, and few-shot performance | `eval_bucket.py` | Markdown and CSV reports |
| Reproduce the manuscript's frequency analysis | `eval_bucket.py` | per-class and overall bucket tables |

For a complete model assessment, use `eval.py` for the conventional prediction
record and `eval_bucket.py` for the stratified report. Because both scripts run
inference independently, use the same dataset, reaction-class setting, beam
size, maximum steps, and intended checkpoint when comparing their results.

## Archived bucket results

The processed USPTO-50K test set contains 5,004 reactions:

- 4,832 many-shot reactions (96.56%);
- 120 medium-shot reactions (2.40%);
- 52 few-shot reactions (1.04%).

The archived G2E-Aug reports contain:

| Reaction-class input | Bucket | n | Top-1 | Top-3 | Top-5 | Top-10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Unknown | Many-shot | 4,832 | 56.6% | 79.3% | 85.9% | 91.2% |
| Unknown | Medium-shot | 120 | 25.8% | 42.5% | 50.8% | 69.2% |
| Unknown | Few-shot | 52 | 19.2% | 32.7% | 42.3% | 50.0% |
| Known | Many-shot | 4,832 | 68.4% | 88.6% | 92.4% | 95.1% |
| Known | Medium-shot | 120 | 46.7% | 62.5% | 72.5% | 82.5% |
| Known | Few-shot | 52 | 17.3% | 46.2% | 51.9% | 57.7% |

Archived reports:

- [Reaction class unknown, epoch 86](experiment/uspto_50k/without_rxn_class/16-07-2026--01-17-31/bucket_eval_epoch_86.md)
- [Reaction class known, epoch 124](experiment/uspto_50k/with_rxn_class/15-07-2026--01-22-06/bucket_eval_epoch_124.md)

## Round-trip option

`eval_bucket.py --round_trip` is an experimental path that requires a separately
configured OpenNMT forward-reaction model and `--rt_model`. It was disabled for
the archived reports and is not needed to reproduce the exact-match results.

## Interpretation and limitations

- The six augmentations were introduced jointly; no component-wise ablation is
  available.
- The model was trained once per reported setting, so run-to-run variance was
  not estimated.
- The energy-aware auxiliary head does not directly rank beam candidates.
- The thresholds 20 and 100 are conventional and were not tuned.
- Only 52 test reactions are few-shot under this definition.
- Exact match counts chemically valid alternatives as incorrect when they differ
  from the recorded reactants.

The repository should therefore be used to study and reproduce the evaluation
protocol, not to attribute an observed difference to one augmentation.

## Upstream project

This work is derived from Graph2Edits:

> Zhong W, Yang Z, Chen CY-C. Retrosynthesis prediction using an end-to-end
> graph generative architecture for molecular graph editing. *Nature
> Communications*. 2023;14:3009.
> <https://doi.org/10.1038/s41467-023-38851-5>

The original Graph2Edits release is archived at
[Zenodo](https://doi.org/10.5281/zenodo.7837349). Please cite the original work
when using the base architecture.

## License

Code in this repository is provided under the [MIT License](LICENSE). USPTO-derived
data and third-party materials remain subject to their respective terms.

## Citation and contact

The bibliographic record for the accompanying manuscript will be added after
publication. For questions about the implementation or evaluation protocol,
please open an issue in this GitHub repository.
