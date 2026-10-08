# G2E-Aug

Code and frequency-stratified evaluation for graph-editing single-step retrosynthesis.

## Title

**Average top-k accuracy conceals a forty-point head-to-tail gap in graph-editing retrosynthesis**

G2E-Aug extends the Graph2Edits pipeline and evaluates retrosynthesis performance by the training frequency of the graph edits required by each test reaction. The repository contains the complete source code for preprocessing, training, beam-search inference, conventional Top-k evaluation, and many-/medium-/few-shot evaluation.

## Overview

The study asks whether aggregate Top-k accuracy hides systematic differences between frequent and rare reaction edits. Two complementary evaluation programs are provided:

- `eval.py` runs conventional beam-search evaluation and saves reaction-level predictions.
- `eval_bucket.py` independently runs beam search and reports Top-1, Top-3, Top-5, and Top-10 exact-match accuracy by edit-frequency bucket.

The two programs do not read each other's outputs. Use the same checkpoint and inference settings with both programs when comparing conventional and stratified results.

Relative to the Graph2Edits backbone, the research branch jointly enables:

1. a tabulated bond-dissociation-energy feature and auxiliary reaction-centre head;
2. a cost-sensitive auxiliary bond-breaking loss;
3. product-reactant graph-representation alignment;
4. effective-number weighting for edit and leaving-group targets;
5. posterior logit adjustment for edit and leaving-group scores; and
6. leaving-group co-occurrence initialization and neighbour regularization.

These components were enabled together. The current release does not include a component-wise ablation and should not be used to attribute an observed change to one component alone.

## Environment Requirements

The validated core environment is:

| Package | Version |
| --- | --- |
| Python | 3.11.8 |
| PyTorch | 2.2.2 |
| NumPy | 1.26.4 |
| RDKit | 2024.03.4 |

The code also uses `pandas`, `joblib`, and `tqdm`. A reference installation is:

```bash
conda create -n g2e-aug python=3.11.8
conda activate g2e-aug
conda install -c conda-forge rdkit=2024.03.4 numpy=1.26.4
python -m pip install torch==2.2.2 pandas joblib tqdm
```

Verify the core versions:

```bash
python -c "import torch, numpy; from rdkit import rdBase; print(torch.__version__, numpy.__version__, rdBase.rdkitVersion)"
```

CUDA is recommended for training and beam-search evaluation. `eval.py` supports loading a checkpoint on the selected CUDA or CPU device. OpenNMT is not required for the reported exact-match results; it is needed only for the optional round-trip path described below.

## Data

### Source and license

The experiments use the USPTO-50K reaction benchmark derived from Lowe's USPTO reaction data. The source collection is available from [Figshare](https://doi.org/10.6084/m9.figshare.5104873.v1) under CC0. The data preparation procedure follows the Graph2Edits representation and then creates the additional frequency and co-occurrence artifacts required by G2E-Aug.

Please cite the original data source and Graph2Edits when using these derived files. USPTO-derived data and other third-party materials remain subject to their respective terms.

### Input splits

Place the canonicalized CSV files in `data/uspto_50k/`:

```text
data/uspto_50k/
  canonicalized_train.csv    40,008 input records
  canonicalized_valid.csv     5,001 input records
  canonicalized_test.csv      5,007 input records
```

The archived evaluation contains 5,004 successfully processed test reactions. Three input test records are excluded during preprocessing because they do not yield valid evaluation examples. This distinction is reported explicitly so that the input-file size is not confused with the evaluation denominator.

### Processed layout

After preprocessing, the relevant files are:

```text
data/uspto_50k/
  train/
    train.file.kekulized
    bond_vocab.txt
    atom_lg_vocab.txt
    action_freq_map.pkl
    lg_cooccurrence_adj.pt
    edit_class_stats.pt
    without_rxn_class/batch-*.pt
    with_rxn_class/batch-*.pt
  valid/
    valid.file.kekulized
  test/
    test.file.kekulized
```

The CSV inputs and generated data artifacts are intentionally excluded from Git because they are dataset artifacts rather than source code. Their permanent archive status is listed under [Reproducibility and availability](#reproducibility-and-availability).

## Data preprocessing

Run all commands from the repository root.

First, extract graph-edit sequences and build the vocabularies, edit-frequency map, leaving-group co-occurrence matrix, and edit-class statistics:

```bash
python preprocess.py --dataset uspto_50k --mode train
python preprocess.py --dataset uspto_50k --mode valid
python preprocess.py --dataset uspto_50k --mode test
```

Then tensorize the training set for both reaction-class settings:

```bash
python prepare_data.py --dataset uspto_50k --mode train
python prepare_data.py --dataset uspto_50k --mode train --use_rxn_class
```

`action_freq_map.pkl` is generated from ground-truth training edits. If it is absent, `eval_bucket.py` attempts to rebuild it from `train.file.kekulized`.

## Train G2E-Aug

Train without reaction-class input:

```bash
python train.py --dataset uspto_50k
```

Train with reaction-class input:

```bash
python train.py --dataset uspto_50k --use_rxn_class
```

Checkpoints and training logs are written to timestamped directories:

```text
experiments/uspto_50k/without_rxn_class/<run-name>/
experiments/uspto_50k/with_rxn_class/<run-name>/
```

The default training configuration uses 200 epochs. All model and loss hyperparameters are recorded in the saved checkpoint under `saveables` and in the experiment log.

## Test

The reported results use beam size 10 and a maximum of 9 graph-edit steps.

### Conventional evaluation: `eval.py`

Reaction class unknown:

```bash
python eval.py --dataset uspto_50k --experiments 16-07-2026--01-17-31 --epoch epoch_86.pt --beam_size 10 --max_steps 9
```

Reaction class known:

```bash
python eval.py --dataset uspto_50k --use_rxn_class --experiments 15-07-2026--01-22-06 --epoch epoch_124.pt --beam_size 10 --max_steps 9
```

`eval.py` reports conventional Top-k exact-match accuracy and writes all beam candidates, probabilities, and edit sequences to `pred_results.txt` in the selected experiment directory. Existing results are preserved by adding a numerical suffix.

### Frequency-stratified evaluation: `eval_bucket.py`

Reaction class unknown:

```bash
python eval_bucket.py --dataset uspto_50k --experiments 16-07-2026--01-17-31 --epoch epoch_86.pt --beam_size 10 --max_steps 9
```

Reaction class known:

```bash
python eval_bucket.py --dataset uspto_50k --use_rxn_class --experiments 15-07-2026--01-22-06 --epoch epoch_124.pt --beam_size 10 --max_steps 9
```

By default, `eval_bucket.py` writes `bucket_eval_epoch_<N>.md` and `bucket_eval_epoch_<N>.csv` in the selected experiment directory. Custom paths can be supplied with `--report_file` and `--csv_file`.

### Bucket definition

For each test reaction, the evaluator finds the least frequent non-terminal ground-truth edit required by that reaction and assigns the entire reaction to one bucket:

| Bucket | Minimum training frequency among required edits |
| --- | ---: |
| Many-shot | greater than 100 |
| Medium-shot | 20 to 100, inclusive |
| Few-shot | less than 20, including unseen edits |

`Terminate` is excluded from frequency assignment. The unit of evaluation is a reaction, not an individual edit. The thresholds are fixed evaluation settings and were not tuned on the test set.

### Which evaluator should be used?

| Goal | Program | Output |
| --- | --- | --- |
| Inspect individual beam predictions | `eval.py` | `pred_results*.txt` |
| Report conventional aggregate Top-k accuracy | `eval.py` | console summary and prediction record |
| Measure many-/medium-/few-shot performance | `eval_bucket.py` | Markdown and CSV reports |
| Reproduce the manuscript frequency analysis | both | conventional and stratified views of the same checkpoint |

## Reproducing our results

Place the processed data and checkpoints at the following paths before running the two evaluation programs:

```text
data/uspto_50k/test/test.file.kekulized
data/uspto_50k/train/action_freq_map.pkl
data/uspto_50k/train/lg_cooccurrence_adj.pt
data/uspto_50k/train/edit_class_stats.pt
experiments/uspto_50k/without_rxn_class/16-07-2026--01-17-31/epoch_86.pt
experiments/uspto_50k/with_rxn_class/15-07-2026--01-22-06/epoch_124.pt
```

Checkpoint verification:

| Setting | Checkpoint | SHA-256 |
| --- | --- | --- |
| Reaction class unknown | `epoch_86.pt` | `540c41d8663faf2f3f16ec6bdb3d658c22e1fbb76667c6a720aee07aa99c0e16` |
| Reaction class known | `epoch_124.pt` | `0522882ab2a1afae03581d5c34ea97081a91cc687df8afabc7075bfec3f9e033` |

The archived stratified results are:

| Reaction-class input | Bucket | n | Top-1 | Top-3 | Top-5 | Top-10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Unknown | Many-shot | 4,832 | 56.6% | 79.3% | 85.9% | 91.2% |
| Unknown | Medium-shot | 120 | 25.8% | 42.5% | 50.8% | 69.2% |
| Unknown | Few-shot | 52 | 19.2% | 32.7% | 42.3% | 50.0% |
| Known | Many-shot | 4,832 | 68.4% | 88.6% | 92.4% | 95.1% |
| Known | Medium-shot | 120 | 46.7% | 62.5% | 72.5% | 82.5% |
| Known | Few-shot | 52 | 17.3% | 46.2% | 51.9% | 57.7% |

Machine-readable and Markdown reports are included in the repository:

- [reaction class unknown, epoch 86](experiment/uspto_50k/without_rxn_class/16-07-2026--01-17-31/bucket_eval_epoch_86.md)
- [reaction class known, epoch 124](experiment/uspto_50k/with_rxn_class/15-07-2026--01-22-06/bucket_eval_epoch_124.md)

## Repository structure

```text
data/
  Atoms_character.txt        atom-property lookup table
  Bond_Energy.txt            bond-energy lookup table
models/
  beam_search.py             beam-search inference
  encoder.py                 graph encoder components
  graph2edits.py             augmented model
utils/
  action_frequencies.py      training-edit frequency counter
  class_balance.py           class weighting and logit adjustment
  chem.py                    chemistry and energy-feature utilities
  collate_fn.py              inference batching
  reaction_actions.py        graph-edit actions
  rxn_graphs.py              molecular graph construction
preprocess.py                extract edit sequences and build artifacts
prepare_data.py              tensorize training edit sequences
train.py                     train G2E-Aug
eval.py                      conventional evaluation
eval_bucket.py               frequency-stratified evaluation
experiment/                  lightweight archived reports and logs
```

The executable scripts use `experiments/` (plural) for checkpoints and generated outputs. The lightweight reports committed to Git are retained under `experiment/` (singular).

## Reproducibility and availability

| Research artifact | Location | Current status |
| --- | --- | --- |
| Complete source code | [GitHub repository](https://github.com/wenju2002-coder/G2E-Aug) | public |
| Archived result tables and logs | `experiment/` in this repository | public |
| Original USPTO source collection | [Figshare DOI](https://doi.org/10.6084/m9.figshare.5104873.v1) | public, CC0 |
| Canonicalized splits and processed artifacts | versioned Zenodo reproducibility archive | DOI to be added before manuscript submission |
| Trained checkpoints | versioned Zenodo reproducibility archive | DOI to be added before manuscript submission |
| Versioned source release | GitHub release connected to Zenodo | DOI to be added before manuscript submission |

The pending DOI entries are deliberately not replaced by invented identifiers. For a journal submission, publish the two Zenodo records, insert their permanent DOI links here and in the manuscript's Data Availability and Code Availability statements, and cite the exact archived version rather than the moving `main` branch.

## Scope and limitations

- All six augmentations were trained jointly; component-wise effects were not isolated.
- One training run is reported for each reaction-class setting, so run-to-run variance is not estimated.
- The auxiliary energy-aware head contributes through training losses but does not directly rank beam candidates.
- Only 52 evaluated reactions are few-shot under the stated thresholds.
- Exact match treats an unrecorded but chemically valid alternative as incorrect.
- Round-trip evaluation was disabled for the archived results.

The optional `eval_bucket.py --round_trip` path requires a separately configured OpenNMT forward-reaction model supplied through `--rt_model`; it is not needed to reproduce the reported exact-match tables.

## Upstream project

This work is derived from Graph2Edits:

> Zhong W, Yang Z, Chen CY-C. Retrosynthesis prediction using an end-to-end graph generative architecture for molecular graph editing. *Nature Communications*. 2023;14:3009. [https://doi.org/10.1038/s41467-023-38851-5](https://doi.org/10.1038/s41467-023-38851-5)

The original implementation is available on [GitHub](https://github.com/Jamson-Zhong/Graph2Edits) and archived on [Zenodo](https://doi.org/10.5281/zenodo.7837349). Please cite the original work when using the base architecture.

## Citation

The bibliographic record for the accompanying manuscript will be added after a preprint or journal DOI is assigned. Until then, cite the versioned G2E-Aug software release together with the Graph2Edits article.

```bibtex
@software{zhang_g2e_aug_2026,
  author  = {Wenju Zhang and Xiaorui Wang and Yachao Cui and Yuquan Li and Pei Yang},
  title   = {G2E-Aug: Frequency-stratified evaluation for graph-editing retrosynthesis},
  year    = {2026},
  url     = {https://github.com/wenju2002-coder/G2E-Aug}
}
```

## License

The source code is released under the [MIT License](LICENSE).

## Contact

For implementation and reproducibility questions, open a GitHub issue or contact Wenju Zhang at `ys250854040484@mailbox.qhu.edu.cn`.
