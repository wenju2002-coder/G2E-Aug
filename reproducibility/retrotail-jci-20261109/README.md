# RetroTail manuscript-matched data and checkpoints

Version: `retrotail-data-20261109` (9 November 2026).

These archives support **RetroTail: Edit-frequency stratified evaluation for
single-step retrosynthesis prediction** by Wenju Zhang, Xiaorui Wang, Pei Yang,
Yachao Cui and Yuquan Li. This is an evaluation workflow and a Graph2Edits-family
case study on one USPTO-50K split, not a cross-family validation or a new
predictor. No new predictions or training runs were generated for this release.

## Download

The files are independent ZIPs, not split volumes. Download each link below
(GitHub's **Download raw file** button also works):

| File | Public download | Contents |
| --- | --- | --- |
| Additional_file_2.zip | [Download](https://github.com/wenju2002-coder/G2E-Aug/raw/refs/tags/retrotail-data-20261109/reproducibility/retrotail-jci-20261109/Additional_file_2.zip) | Matched predictions, statistics, analysis and plotting source |
| Additional_file_3.zip | [Download](https://github.com/wenju2002-coder/G2E-Aug/raw/refs/tags/retrotail-data-20261109/reproducibility/retrotail-jci-20261109/Additional_file_3.zip) | Baseline source and reaction data |
| Additional_file_4.zip | [Download](https://github.com/wenju2002-coder/G2E-Aug/raw/refs/tags/retrotail-data-20261109/reproducibility/retrotail-jci-20261109/Additional_file_4.zip) | Baseline weights, logs and raw prediction records |
| Additional_file_5.zip | [Download](https://github.com/wenju2002-coder/G2E-Aug/raw/refs/tags/retrotail-data-20261109/reproducibility/retrotail-jci-20261109/Additional_file_5.zip) | Augmented source and reaction data |
| Additional_file_6.zip | [Download](https://github.com/wenju2002-coder/G2E-Aug/raw/refs/tags/retrotail-data-20261109/reproducibility/retrotail-jci-20261109/Additional_file_6.zip) | Augmented weights, logs and raw prediction records |

The ZIP files are byte-identical to the locally prepared manuscript Additional
files 2-6. Their SHA-256 values and contained-file checks are recorded in
`SHA256SUMS.json` and `EVIDENCE_VALIDATION.json`. Do not treat a filename alone as
a version identifier.

## Extract and use

Extract all five ZIPs into the **same empty parent directory**, preserving the
directory structure. Do not rename `experiment/` (Aug) to `experiments/`
(baseline), or mix model weights between the two implementations.

```text
Analysis/          four matched per-reaction CSVs and aggregate analysis tables
evidence_analysis/ statistical/plotting source and inputs, figures, provenance
Graph2Edits/       archived baseline source, data, checkpoints and predictions
Graph2Edits-Aug/   archived augmented source, data, checkpoints and predictions
```

The four `Analysis/*_per_reaction.csv` tables contain the same 5,004 test reactions,
with 4,832 many-shot, 120 medium-shot and 52 few-shot reactions each. They retain
ranked top-10 candidate records, including correctness flags. Summary tables do
not replace these matched records for paired comparisons.

| Model | Reaction-class input | Checkpoint | Retained raw prediction file |
| --- | --- | --- | --- |
| Graph2Edits | Withheld | `epoch_123.pt` | `pred_results.txt` |
| Graph2Edits | Supplied | `epoch_92.pt` | `pred_results.txt` |
| Graph2Edits-Aug | Withheld | `epoch_86.pt` | `pred_results_1.txt` |
| Graph2Edits-Aug | Supplied | `epoch_124.pt` | `pred_results_4.txt` |

Use each extracted model directory's README for its historical software
environment and evaluator commands. The current repository-root README describes
the maintained Aug evaluator. Statistical re-analysis of retained predictions
does not require retraining or re-running beam search. Historical environment
differences and unrecorded random seeds remain limitations; packaging this
archive does not establish fresh-environment or multi-seed reproducibility.

The ZIPs retain historical layout folders, manifests and README statements for
provenance. In particular, internal statements that GitHub does not contain data
or checkpoints predate this public deposit and are superseded by **this outer
README only for availability**. They do not change the numerical evidence.
`README_DMKD.md`, older figure scripts and `ESM_1.pdf` are historical names, not
new experiments or additional manuscripts to submit. The current Figure 2 source
is `evidence_analysis/approved_figure2_20261109/`; the current merged Figure 4
source is `evidence_analysis/approved_figure4/`. The supplied final figure assets
are in `evidence_analysis/figures/`. Table S5 describes hypothetical power
planning, not observed improvements.

## Licence and provenance

New analysis/plotting code is MIT-licensed; newly prepared statistical tables and
curated prediction records are CC BY 4.0. See `LICENSE_SCOPE_JCI.md` and
`LICENSE_RetroTail_Code.txt`. Original data, model checkpoints, third-party
figures and upstream source retain their original notices and terms; the new
licences do not relicense those materials. Baseline and Aug software directories
retain the original Graph2Edits MIT notice.

Original Graph2Edits:

- [Source](https://github.com/Jamson-Zhong/Graph2Edits)
- [Original software archive](https://doi.org/10.5281/zenodo.7837349)
- Zhong W, Yang Z, Chen CY-C (2023). Retrosynthesis prediction using an end-to-end
  graph generative architecture for molecular graph editing. Nature
  Communications 14:3009. https://doi.org/10.1038/s41467-023-38851-5

USPTO source corpus: https://doi.org/10.6084/m9.figshare.5104873.v1.
The original Graph2Edits DOI is not a DOI for this RetroTail deposit. This release
has a Git tag and file hashes; no study-specific Zenodo DOI is claimed.

Checkpoint, pickle and joblib formats may execute code when loaded. Verify
provenance and hashes first, and only load files you trust. No environments,
credentials, main manuscript, author title page or Cover Letter are included as
standalone repository files by this deposit.
