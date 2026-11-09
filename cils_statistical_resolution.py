"""Reproduce the manuscript's approximate unpaired power-planning calculations.

MDD is searched on a 0.1-percentage-point grid. It is a design calculation,
not a significance test on the observed paired predictions.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import NormalDist


def minimum_detectable_difference(n: int, p0: float) -> float:
    z_alpha = NormalDist().inv_cdf(0.975)
    z_power = NormalDist().inv_cdf(0.80)
    for step in range(1, 1001):
        delta = step / 1000
        p1 = p0 + delta
        if p1 > 1:
            break
        pooled = (p0 + p1) / 2
        requirement = (z_alpha * math.sqrt(2 * pooled * (1 - pooled) / n)
                       + z_power * math.sqrt((p0 * (1 - p0) + p1 * (1 - p1)) / n))
        if delta >= requirement:
            return round(delta * 100, 1)
    return float('nan')


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    records = [
        ('Unknown', 'Many-shot', 4832, 56.2, 90.4),
        ('Unknown', 'Medium-shot', 120, 100 * 35 / 120, 100 * 81 / 120),
        ('Unknown', 'Few-shot', 52, 100 * 7 / 52, 100 * 20 / 52),
        ('Known', 'Many-shot', 4832, 68.1, 94.5),
        ('Known', 'Medium-shot', 120, 100 * 58 / 120, 100 * 98 / 120),
        ('Known', 'Few-shot', 52, 100 * 12 / 52, 100 * 30 / 52),
    ]
    rows = []
    for setting, bucket, n, top1, top10 in records:
        rows.append({'setting': setting, 'bucket': bucket, 'n_per_group': n,
                     'baseline_top1_percent': round(top1, 4),
                     'baseline_top10_percent': round(top10, 4),
                     'top1_mdd_pp': minimum_detectable_difference(n, top1 / 100),
                     'top10_mdd_pp': minimum_detectable_difference(n, top10 / 100),
                     'aggregate_top1_ceiling_pp': round(n / 5004 * (100 - top1), 4),
                     'aggregate_top10_ceiling_pp': round(n / 5004 * (100 - top10), 4)})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == '__main__':
    main()
