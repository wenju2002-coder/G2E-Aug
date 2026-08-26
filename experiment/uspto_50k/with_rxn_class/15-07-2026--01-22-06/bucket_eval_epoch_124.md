# Bucket Evaluation: 15-07-2026--01-22-06

- Dataset: `uspto_50k`
- Checkpoint: `epoch_124.pt`
- Beam size: `10`
- Max steps: `9`
- Unit: `reaction`
- Test reactions: `5004`
- Use reaction class: `True`
- Round-trip: `False`

## Evaluation Report (Bucket by Minimum Edit Frequency)

| Reaction Class | Bucket | Total Reactions | Top-1 | Top-3 | Top-5 | Top-10 | Round-trip |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Many-shot | 1513 | 66.8% | 87.2% | 92.1% | 95.4% | - |
| 0 | Medium-shot | 2 | 50.0% | 50.0% | 50.0% | 50.0% | - |
| 0 | Few-shot | 1 | 100.0% | 100.0% | 100.0% | 100.0% | - |
| 1 | Many-shot | 1173 | 77.1% | 95.1% | 96.7% | 97.6% | - |
| 1 | Medium-shot | 4 | 100.0% | 100.0% | 100.0% | 100.0% | - |
| 1 | Few-shot | 10 | 20.0% | 80.0% | 80.0% | 80.0% | - |
| 2 | Many-shot | 507 | 53.8% | 74.8% | 80.9% | 88.2% | - |
| 2 | Medium-shot | 49 | 38.8% | 61.2% | 69.4% | 81.6% | - |
| 2 | Few-shot | 11 | 18.2% | 27.3% | 27.3% | 27.3% | - |
| 3 | Many-shot | 83 | 69.9% | 83.1% | 84.3% | 89.2% | - |
| 3 | Medium-shot | 5 | 80.0% | 80.0% | 80.0% | 80.0% | - |
| 3 | Few-shot | 3 | 66.7% | 100.0% | 100.0% | 100.0% | - |
| 4 | Many-shot | 61 | 93.4% | 98.4% | 98.4% | 100.0% | - |
| 4 | Medium-shot | 7 | 100.0% | 100.0% | 100.0% | 100.0% | - |
| 5 | Many-shot | 782 | 59.7% | 89.0% | 93.4% | 95.3% | - |
| 5 | Medium-shot | 37 | 37.8% | 51.4% | 67.6% | 81.1% | - |
| 5 | Few-shot | 5 | 0.0% | 40.0% | 40.0% | 40.0% | - |
| 6 | Many-shot | 445 | 73.9% | 91.2% | 93.9% | 96.6% | - |
| 6 | Medium-shot | 5 | 0.0% | 40.0% | 40.0% | 60.0% | - |
| 6 | Few-shot | 12 | 8.3% | 25.0% | 33.3% | 50.0% | - |
| 7 | Many-shot | 81 | 88.9% | 95.1% | 97.5% | 97.5% | - |
| 7 | Few-shot | 1 | 100.0% | 100.0% | 100.0% | 100.0% | - |
| 8 | Many-shot | 164 | 70.1% | 84.8% | 89.6% | 92.1% | - |
| 8 | Medium-shot | 11 | 63.6% | 72.7% | 90.9% | 90.9% | - |
| 8 | Few-shot | 9 | 0.0% | 33.3% | 55.6% | 66.7% | - |
| 9 | Many-shot | 23 | 91.3% | 95.7% | 95.7% | 95.7% | - |

### Overall Summary (All Reaction Classes)

| Bucket | Total Reactions | Top-1 | Top-3 | Top-5 | Top-10 | Round-trip |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Many-shot | 4832 | 68.4% | 88.6% | 92.4% | 95.1% | - |
| Medium-shot | 120 | 46.7% | 62.5% | 72.5% | 82.5% | - |
| Few-shot | 52 | 17.3% | 46.2% | 51.9% | 57.7% | - |
