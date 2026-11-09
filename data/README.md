# Published Graph2Edits-Aug data

The original `Atoms_character.txt` and `Bond_Energy.txt` lookups are retained.
`uspto_50k/` directly contains all six original raw/canonicalised CSVs, plus the
manuscript-matched processed reactions, vocabularies and training-frequency map.
No new preprocessing or predictions were generated for this release.

| Split | Raw CSV | Canonicalised CSV | Processed records |
| --- | --- | --- | --- |
| Training | `uspto_50k/raw_train.csv` | `uspto_50k/canonicalized_train.csv` | `uspto_50k/train/train.file.kekulized` |
| Validation | `uspto_50k/raw_valid.csv` | `uspto_50k/canonicalized_valid.csv` | `uspto_50k/valid/valid.file.kekulized` |
| Test | `uspto_50k/raw_test.csv` | `uspto_50k/canonicalized_test.csv` | `uspto_50k/test/test.file.kekulized` |

The processed split has 39,985 training and 5,004 test reactions. The canonicalised
test file has 5,007 reactions; three with unsupported Add Bond edits are excluded
by preprocessing. The test-frequency strata contain 4,832 many-shot, 120
medium-shot and 52 few-shot reactions. Class indices in stored records are 0-9;
the paper's standard class numbers are those indices plus one.

Processed files are serialized Python/joblib artifacts, not plain-text CSVs.
Only deserialize trusted files and use the corresponding model environment.
Training tensor batches can be generated with `prepare_data.py`; they are not
needed to run the retained checkpoint evaluator and are not claimed to be
pre-generated in this release. Historical seeds are not available.

The source corpus is Lowe's USPTO collection:
https://doi.org/10.6084/m9.figshare.5104873.v1. The split and preprocessing derive
from [Graph2Edits](https://github.com/Jamson-Zhong/Graph2Edits), whose original
MIT notice is retained in the repository LICENSE. Original data and serialized
artifacts retain their upstream provenance and terms; the software licence does
not purport to relicense third-party data or grant ownership of patent records.
See the root `DATA_CHECKPOINTS_SHA256.json` for exact hashes and sources.
