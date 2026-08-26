# Bucket Evaluation: 16-07-2026--01-17-31

- Dataset: `uspto_50k`
- Checkpoint: `epoch_86.pt`
- Beam size: `10`
- Max steps: `9`
- Unit: `reaction`
- Test reactions: `5004`
- Use reaction class: `False`
- Round-trip: `False`

## Evaluation Report (Bucket by Minimum Edit Frequency)

| Reaction Class | Bucket | Total Reactions | Top-1 | Top-3 | Top-5 | Top-10 | Round-trip |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Many-shot | 1513 | 55.7% | 80.0% | 86.6% | 91.9% | - |
| 0 | Medium-shot | 2 | 50.0% | 50.0% | 50.0% | 50.0% | - |
| 0 | Few-shot | 1 | 100.0% | 100.0% | 100.0% | 100.0% | - |
| 1 | Many-shot | 1173 | 67.4% | 89.3% | 93.7% | 96.8% | - |
| 1 | Medium-shot | 4 | 100.0% | 100.0% | 100.0% | 100.0% | - |
| 1 | Few-shot | 10 | 40.0% | 70.0% | 80.0% | 90.0% | - |
| 2 | Many-shot | 507 | 45.0% | 63.5% | 72.6% | 81.3% | - |
| 2 | Medium-shot | 49 | 22.4% | 34.7% | 38.8% | 61.2% | - |
| 2 | Few-shot | 11 | 9.1% | 27.3% | 36.4% | 36.4% | - |
| 3 | Many-shot | 83 | 47.0% | 66.3% | 74.7% | 77.1% | - |
| 3 | Medium-shot | 5 | 40.0% | 60.0% | 60.0% | 80.0% | - |
| 3 | Few-shot | 3 | 0.0% | 33.3% | 33.3% | 66.7% | - |
| 4 | Many-shot | 61 | 65.6% | 88.5% | 93.4% | 96.7% | - |
| 4 | Medium-shot | 7 | 100.0% | 100.0% | 100.0% | 100.0% | - |
| 5 | Many-shot | 782 | 53.6% | 79.5% | 86.8% | 92.3% | - |
| 5 | Medium-shot | 37 | 13.5% | 37.8% | 54.1% | 73.0% | - |
| 5 | Few-shot | 5 | 20.0% | 20.0% | 60.0% | 80.0% | - |
| 6 | Many-shot | 445 | 54.8% | 75.5% | 83.8% | 89.9% | - |
| 6 | Medium-shot | 5 | 0.0% | 20.0% | 20.0% | 20.0% | - |
| 6 | Few-shot | 12 | 16.7% | 16.7% | 16.7% | 25.0% | - |
| 7 | Many-shot | 81 | 63.0% | 86.4% | 90.1% | 93.8% | - |
| 7 | Few-shot | 1 | 100.0% | 100.0% | 100.0% | 100.0% | - |
| 8 | Many-shot | 164 | 38.4% | 58.5% | 65.2% | 78.0% | - |
| 8 | Medium-shot | 11 | 9.1% | 36.4% | 54.5% | 81.8% | - |
| 8 | Few-shot | 9 | 0.0% | 11.1% | 22.2% | 22.2% | - |
| 9 | Many-shot | 23 | 73.9% | 87.0% | 91.3% | 91.3% | - |

### Overall Summary (All Reaction Classes)

| Bucket | Total Reactions | Top-1 | Top-3 | Top-5 | Top-10 | Round-trip |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Many-shot | 4832 | 56.6% | 79.3% | 85.9% | 91.2% | - |
| Medium-shot | 120 | 25.8% | 42.5% | 50.8% | 69.2% | - |
| Few-shot | 52 | 19.2% | 32.7% | 42.3% | 50.0% | - |
