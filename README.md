# G2E-Aug

> **论文投稿中，为避免版权和复制风险，我们建议中稿后补全并公开，有问题请联系我们。**

This repository is the public, review-period snapshot of **Graph2Edits-Aug**, a
research project for frequency-stratified evaluation of graph-editing
retrosynthesis. The implementation will be completed after the associated
manuscript is accepted.

## Release status

The repository intentionally preserves the project layout, evaluation protocol,
small chemistry lookup tables, license, and aggregate experiment reports. The
following implementation files are placeholders in this review-period release:

- `models/graph2edits.py` — augmented model implementation;
- `train.py` — optimization and auxiliary-loss implementation;
- `preprocess.py` — training-artifact construction;
- `prepare_data.py` — augmented training-data tensorization;
- `utils/class_balance.py` — long-tail balancing and logit adjustment;
- `utils/chem.py` — chemistry and bond-energy feature helpers.

Because these modules are withheld, this snapshot is **not currently runnable
for training or inference**. Raw USPTO-derived data, generated preprocessing
artifacts, Python bytecode, and model checkpoints are also intentionally absent.
No complete implementation is present in this repository's Git history.

## Available material

- `eval_bucket.py` contains the reaction-level frequency-stratified evaluation
  protocol used to report many-shot, medium-shot, and few-shot exact-match
  accuracy.
- `utils/action_frequencies.py` contains the edit-frequency counting helper.
- `experiment/` contains small aggregate logs and bucket reports, but no model
  weights.
- `data/Atoms_character.txt` and `data/Bond_Energy.txt` are small lookup tables.

The public results distinguish reactions by the minimum training frequency of
their required non-terminal edit actions:

- many-shot: frequency greater than 100;
- medium-shot: frequency from 20 through 100;
- few-shot: frequency below 20.

## Upstream project

This work is derived from Graph2Edits:

> Zhong W, Yang Z, Chen CY-C. Retrosynthesis prediction using an end-to-end
> graph generative architecture for molecular graph editing. *Nature
> Communications*. 2023;14:3009.
> https://doi.org/10.1038/s41467-023-38851-5

Please cite the original work when using the base architecture. Citation details
for the accompanying manuscript will be added after publication.

## License

Code in this snapshot is provided under the [MIT License](LICENSE). Dataset and
third-party materials remain subject to their respective terms.

## Contact

For questions or academic discussion, please contact the repository owner through
GitHub. Full implementation details are planned for release after acceptance.
