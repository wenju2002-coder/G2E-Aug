from __future__ import division, unicode_literals
import argparse
import os

import joblib
import numpy as np
import torch
from rdkit import Chem, RDLogger
from tqdm import tqdm

from models import Graph2Edits, BeamSearch
from utils.class_balance import load_lg_cooccurrence_adj, load_or_build_edit_class_stats

lg = RDLogger.logger()
lg.setLevel(4)

ROOT_DIR = './'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_state(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'Checkpoint missing newly added parameters: {missing}')
    if unexpected:
        print(f'Checkpoint has unexpected parameters: {unexpected}')


def canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
    except Exception:
        return smi
    if mol is None:
        return smi
    mol = Chem.RemoveHs(mol)
    [atom.ClearProp('molAtomMapNumber') for atom in mol.GetAtoms()]
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def smi_tokenizer(smi):
    import re
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    if smi != ''.join(tokens):
        return None
    return ' '.join(tokens)


def get_experiment_dir(args):
    class_dir = 'with_rxn_class' if args.use_rxn_class else 'without_rxn_class'
    return os.path.join(ROOT_DIR, 'experiments', args.dataset, class_dir, args.experiments)


def load_graph2edits_model(args, exp_dir):
    checkpoint = torch.load(os.path.join(exp_dir, args.epoch))
    config = checkpoint['saveables']

    model = Graph2Edits(**config, device=DEVICE)
    load_model_state(model, checkpoint['state'])
    model.to(DEVICE)

    dataset_root = os.path.join(ROOT_DIR, 'data', args.dataset)
    model.set_lg_cooccurrence_adj(
        load_lg_cooccurrence_adj(
            os.path.join(dataset_root, 'train', 'lg_cooccurrence_adj.pt'),
            len(config['atom_vocab'])))
    model.set_edit_class_statistics(
        load_or_build_edit_class_stats(dataset_root, config['bond_vocab'], config['atom_vocab']),
        initialize_lg_tail=False)
    model.eval()
    return model


def load_forward_predictions(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as fp:
        return [''.join(line.strip().split(' ')) for line in fp.readlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='uspto_50k',
                        help='dataset: uspto_50k or uspto_full')
    parser.add_argument('--use_rxn_class', default=False, action='store_true',
                        help='Whether to use reaction class')
    parser.add_argument('--experiments', type=str, default='10-07-2026--08-59-05',
                        help='Name of edits prediction experiment')
    parser.add_argument('--epoch', type=str, default='epoch_71.pt',
                        help='Checkpoint file name, e.g. epoch_123.pt')
    parser.add_argument('--beam_size', type=int, default=50,
                        help='Beam search width')
    parser.add_argument('--max_steps', type=int, default=9,
                        help='Maximum number of edit steps')
    parser.add_argument('--forward_inputs', type=str, default=None,
                        help='Forward-model input file written under the experiment directory')
    parser.add_argument('--forward_predictions', type=str, default=None,
                        help='Forward-model output file read under the experiment directory')
    args = parser.parse_args()
    args.dataset = args.dataset.lower()

    exp_dir = get_experiment_dir(args)
    pred_text_dir = os.path.join(exp_dir, 'pred_text1')
    os.makedirs(pred_text_dir, exist_ok=True)

    top_tag = f'top{args.beam_size}'
    forward_inputs = args.forward_inputs or f'forward_inputs_50k_{top_tag}.txt'
    forward_predictions = args.forward_predictions or f'forward_predictions_50k_{top_tag}.txt'
    forward_inputs_path = os.path.join(exp_dir, forward_inputs)
    forward_predictions_path = os.path.join(exp_dir, forward_predictions)

    forward_targets = load_forward_predictions(forward_predictions_path)
    if forward_targets is None:
        print(f'Forward predictions not found: {forward_predictions_path}')
        print(f'This run will write forward-model inputs to: {forward_inputs_path}')
    else:
        print(f'Loaded {len(forward_targets)} forward predictions from: {forward_predictions_path}')

    data_dir = os.path.join(ROOT_DIR, 'data', args.dataset, 'test')
    test_data = joblib.load(os.path.join(data_dir, 'test.file.kekulized'))

    model = load_graph2edits_model(args, exp_dir)
    beam_model = BeamSearch(
        model=model, step_beam_size=10, beam_size=args.beam_size,
        use_rxn_class=args.use_rxn_class)

    top_k = np.zeros(args.beam_size)
    rt_top_k = np.zeros(args.beam_size)
    current_target_idx = 0

    p_bar = tqdm(list(range(len(test_data))))
    with open(forward_inputs_path, 'w') as forward_fp:
        for idx in p_bar:
            rxn_data = test_data[idx]
            rxn_smi = rxn_data.rxn_smi
            rxn_class = rxn_data.rxn_class

            reactants_smi, product_smi = rxn_smi.split('>>')
            true_reactants = set(canonicalize(reactants_smi).split('.'))
            canonical_product = canonicalize(product_smi)

            with torch.no_grad():
                top_k_results = beam_model.run_search(
                    prod_smi=product_smi, max_steps=args.max_steps,
                    rxn_class=rxn_class if args.use_rxn_class else None)

            beam_matched = False
            true_idx = None
            valid_beam_indices = []
            pred_text_path = os.path.join(pred_text_dir, f'{idx}.txt')

            with open(pred_text_path, 'w') as pred_fp:
                for beam_idx, path in enumerate(top_k_results):
                    pred_smi = path['final_smi']
                    prob = path.get('prob', 1.0)

                    if pred_smi != 'final_smi_unmapped' and prob > 0.0:
                        tokenized = smi_tokenizer(pred_smi)
                        if tokenized is not None:
                            pred_fp.write(f'{tokenized}\n')
                            forward_fp.write(f'{tokenized}\n')
                            valid_beam_indices.append(beam_idx)

                    pred_set = set(pred_smi.split('.'))
                    if pred_set == true_reactants and not beam_matched:
                        top_k[beam_idx] += 1
                        beam_matched = True
                        true_idx = beam_idx

            end_idx = current_target_idx + len(valid_beam_indices)

            if forward_targets is not None:
                if end_idx > len(forward_targets):
                    raise ValueError(
                        f'Forward prediction file is too short at test index {idx}: '
                        f'needed {end_idx}, got {len(forward_targets)}. '
                        'Regenerate it from the current forward input file.')

                all_predictions = forward_targets[current_target_idx:end_idx]
                rt_matched = False

                for local_idx, prediction in enumerate(all_predictions):
                    beam_idx = valid_beam_indices[local_idx]
                    pred_product = canonicalize(prediction)

                    if beam_matched and not rt_matched:
                        if true_idx <= beam_idx:
                            rt_top_k[true_idx] += 1
                            rt_matched = True
                        elif pred_product == canonical_product:
                            rt_top_k[beam_idx] += 1
                            rt_matched = True

                    if pred_product == canonical_product and not beam_matched and not rt_matched:
                        rt_top_k[beam_idx] += 1
                        rt_matched = True

            current_target_idx = end_idx

            msg = ''
            for beam_idx in [1, 3, 5, 10, 50]:
                if beam_idx > args.beam_size:
                    continue
                exact_acc = np.sum(top_k[:beam_idx]) / (idx + 1)
                msg += 'Exact accuracy, t%d: %.3f' % (beam_idx, exact_acc)
                if forward_targets is not None:
                    rt_acc = np.sum(rt_top_k[:beam_idx]) / (idx + 1)
                    msg += ' Round-trip accuracy, t%d: %.3f ' % (beam_idx, rt_acc)
            p_bar.set_description(msg)

    if forward_targets is None:
        print()
        print('Forward inputs are ready. Translate them with a reactants-to-product forward model, for example:')
        print(f'python translate.py -model /path/to/forward_model.pt -src {forward_inputs_path} '
              f'-output {forward_predictions_path} -gpu 0 -batch_size 64 '
              '-replace_unk -max_length 200 -beam_size 1 -n_best 1')
    elif current_target_idx != len(forward_targets):
        print(f'Warning: used {current_target_idx} forward predictions, '
              f'but file contains {len(forward_targets)} lines.')


if __name__ == '__main__':
    main()
