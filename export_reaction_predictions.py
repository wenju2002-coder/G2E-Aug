"""Export reaction-level predictions using a specified code release and checkpoint.

Run in the scientific environment used for that model, after its preprocessing.
The exporter never resets saved frequency, co-occurrence, or logit buffers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--test-file', type=Path, required=True)
    parser.add_argument('--frequency-map', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-label', required=True)
    parser.add_argument('--use-rxn-class', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    for path in [args.code_root, args.checkpoint, args.test_file, args.frequency_map]:
        if not path.exists():
            parser.error(f'Missing required input: {path}')
    if args.output.exists():
        parser.error(f'Refusing to overwrite predictions: {args.output}')
    sys.path.insert(0, str(args.code_root.resolve()))

    import joblib
    import torch
    from rdkit import Chem, RDLogger
    from models import Graph2Edits, BeamSearch

    RDLogger.DisableLog('rdApp.*')
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = Graph2Edits(**checkpoint['saveables'], device=args.device)
    model.load_state_dict(checkpoint['state'], strict=True)
    model.to(args.device)
    model.eval()
    search = BeamSearch(model=model, step_beam_size=10, beam_size=10,
                        use_rxn_class=args.use_rxn_class)
    records = joblib.load(args.test_file)
    frequencies = joblib.load(args.frequency_map)
    if len(records) != 5004:
        raise ValueError(f'Expected 5004 evaluated reactions; received {len(records)}')

    def unmapped(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol, isomericSmiles=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ['reaction_index', 'reaction_id', 'class_index_0_based', 'class_number_1_based',
              'bucket', 'product_mapped', 'ground_truth_reactants', 'predictions_top10_json',
              'first_match_rank', 'top1', 'top3', 'top5', 'top10', 'model', 'checkpoint_sha256']
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    with args.output.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(records):
            reactants, product = record.rxn_smi.split('>>')
            truth = unmapped(reactants)
            if truth is None:
                raise ValueError(f'Invalid reference reactants at reaction {index}')
            rarest = min((frequencies.get(edit, 0) for edit in record.edits
                          if edit != 'Terminate'), default=float('inf'))
            bucket = 'Many-shot' if rarest > 100 else 'Medium-shot' if rarest >= 20 else 'Few-shot'
            with torch.no_grad():
                paths = search.run_search(prod_smi=product, max_steps=9,
                                          rxn_class=record.rxn_class if args.use_rxn_class else None)
            # Keep rank positions, including failed or invalid candidates.
            predictions = [unmapped(path.get('final_smi', ''))
                           if path.get('final_smi') not in [None, 'final_smi_unmapped'] else None
                           for path in paths[:10]]
            truth_set = set(truth.split('.'))
            match = next((rank for rank, pred in enumerate(predictions, 1)
                          if pred is not None and set(pred.split('.')) == truth_set), None)
            writer.writerow({
                'reaction_index': index, 'reaction_id': getattr(record, 'rxn_id', ''),
                'class_index_0_based': record.rxn_class,
                'class_number_1_based': record.rxn_class + 1,
                'bucket': bucket, 'product_mapped': product, 'ground_truth_reactants': truth,
                'predictions_top10_json': json.dumps(predictions, ensure_ascii=False),
                'first_match_rank': match or '',
                **{f'top{k}': int(match is not None and match <= k) for k in [1, 3, 5, 10]},
                'model': args.model_label, 'checkpoint_sha256': checkpoint_hash,
            })
            if (index + 1) % 50 == 0:
                handle.flush()
                print(f'Exported {index + 1}/{len(records)} reactions', flush=True)
    print(args.output)


if __name__ == '__main__':
    main()
