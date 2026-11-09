from __future__ import division, unicode_literals
import numpy as np
import pandas as pd
import os
import argparse
import joblib
from tqdm import tqdm
import torch
from rdkit import Chem, RDLogger

from models import Graph2Edits, BeamSearch
from utils.class_balance import load_lg_cooccurrence_adj, load_or_build_edit_class_stats
lg = RDLogger.logger()
lg.setLevel(4)


from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator

import onmt.inputters
import onmt.translate
import onmt
import onmt.model_builder
import onmt.modules
import onmt.opts

ROOT_DIR = './'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_RT_MODEL = 'model.pt'

def load_model_state(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'Checkpoint missing newly added parameters: {missing}')
    if unexpected:
        print(f'Checkpoint has unexpected parameters: {unexpected}')


def canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        print('no mol', flush=True)
        return smi
    if mol is None:
        return smi
    mol = Chem.RemoveHs(mol)
    [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]
    return Chem.MolToSmiles(mol)

def canonicalize_p(smi):
    p = canonicalize(smi)
    p_mol = Chem.MolFromSmiles(p)
    [a.SetAtomMapNum(a.GetIdx()+1) for a in p_mol.GetAtoms()]
    p_smi = Chem.MolToSmiles(p_mol)
    return p_smi


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern =  r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def build_round_trip_options(pred_text, args):
    parse = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    if hasattr(onmt.opts, 'add_md_help_argument'):
        onmt.opts.add_md_help_argument(parse)
    onmt.opts.translate_opts(parse)

    extra_onmt_args = args.onmt_args or []
    has_model_arg = any(
        opt == '-model' or opt == '--model' or opt.startswith('-model=') or opt.startswith('--model=')
        for opt in extra_onmt_args
    )

    opt_args = ['-src', pred_text]
    if args.rt_model and not has_model_arg:
        opt_args += ['-model', args.rt_model]
    if args.rt_gpu is not None:
        opt_args += ['-gpu', str(args.rt_gpu)]
    if args.rt_batch_size is not None:
        opt_args += ['-batch_size', str(args.rt_batch_size)]
    opt_args += extra_onmt_args

    opt, unknown = parse.parse_known_args(opt_args)
    if unknown:
        print(f'Warning: OpenNMT ignored unknown args: {unknown}')
    return opt


def get_onmt_model_arg(args):
    extra_onmt_args = args.onmt_args or []
    for idx, opt in enumerate(extra_onmt_args):
        if opt in ('-model', '--model') and idx + 1 < len(extra_onmt_args):
            return extra_onmt_args[idx + 1]
        if opt.startswith('-model=') or opt.startswith('--model='):
            return opt.split('=', 1)[1]
    return args.rt_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='USPTO_50k', help='dataset: USPTO_50k or USPTO_full')
    parser.add_argument("--use_rxn_class", default=True,
                        action='store_true', help='Whether to use rxn_class')
    parser.add_argument('--experiments', type=str, default='12-07-2026--14-22-37', help='Name of edits prediction experiment')
    parser.add_argument('--epoch', type=str, default='epoch_69.pt',
                        help='Checkpoint file name, e.g. epoch_123.pt')
    parser.add_argument('--beam_size', type=int, default=50, help='Beam search width')
    parser.add_argument('--max_steps', type=int, default=9, help='maximum number of edit steps')
    parser.add_argument('--no_round_trip', action='store_true',
                        help='Disable round-trip evaluation and report exact beam accuracy only')
    parser.add_argument('--rt_model', type=str, default=DEFAULT_RT_MODEL,
                        help='Path to the OpenNMT round-trip model; defaults to model.pt like the original script')
    parser.add_argument('--rt_gpu', type=int, default=0,
                        help='GPU id for OpenNMT round-trip model; use -1 for CPU')
    parser.add_argument('--rt_batch_size', type=int, default=50,
                        help='Batch size for OpenNMT round-trip translation')
    parser.add_argument('--onmt_args', nargs=argparse.REMAINDER, default=None,
                        help='Extra OpenNMT translate options, placed after --onmt_args')

    args = parser.parse_args()
    args.dataset = args.dataset.lower()

    if not args.no_round_trip:
        rt_model = get_onmt_model_arg(args)
        if rt_model and not os.path.exists(rt_model):
            raise FileNotFoundError(
                f'Round-trip OpenNMT model not found: {rt_model}. '
                f'The original script assumes {DEFAULT_RT_MODEL} exists in the current directory. '
                'Pass the real forward-model checkpoint with --rt_model /path/to/model.pt, '
                'or run with --no_round_trip for exact-match evaluation only.'
            )

    data_dir = os.path.join(ROOT_DIR, 'data', f'{args.dataset}', 'test')
    test_file = os.path.join(data_dir, 'test.file.kekulized')
    test_data = joblib.load(test_file)
    if args.use_rxn_class:
        exp_dir = os.path.join(ROOT_DIR, 'experiments', f'{args.dataset}', 'with_rxn_class', f'{args.experiments}')
    else:
        exp_dir = os.path.join(ROOT_DIR, 'experiments', f'{args.dataset}', 'without_rxn_class', f'{args.experiments}')

    checkpoint = torch.load(os.path.join(exp_dir, args.epoch))
    config = checkpoint['saveables']
    
    model = Graph2Edits(**config, device=DEVICE)
    load_model_state(model, checkpoint['state'])
    model.to(DEVICE)
    dataset_root = os.path.join(ROOT_DIR, 'data', args.dataset)
    model.set_lg_cooccurrence_adj(
        load_lg_cooccurrence_adj(os.path.join(dataset_root, 'train', 'lg_cooccurrence_adj.pt'),
                                 len(config['atom_vocab'])))
    model.set_edit_class_statistics(
        load_or_build_edit_class_stats(dataset_root, config['bond_vocab'], config['atom_vocab']),
        initialize_lg_tail=False)
    model.eval()

    top_k = np.zeros(args.beam_size)
    rt_top_k = np.zeros(args.beam_size)
    beam_model = BeamSearch(model=model, step_beam_size=10, beam_size=args.beam_size, use_rxn_class=args.use_rxn_class)
    p_bar = tqdm(list(range(len(test_data))))
    pred_text_dir = os.path.join(exp_dir, 'pred_text1')
    os.makedirs(pred_text_dir, exist_ok=True)

    for idx in p_bar:
        rxn_data = test_data[idx]
        rxn_smi = rxn_data.rxn_smi
        rxn_class = rxn_data.rxn_class

        r, p = rxn_smi.split('>>')
        r_mol = Chem.MolFromSmiles(r)
        [a.ClearProp('molAtomMapNumber') for a in r_mol.GetAtoms()]
        r_mol = Chem.MolFromSmiles(Chem.MolToSmiles(r_mol))
        r_smi = Chem.MolToSmiles(r_mol)
        r_set = set(r_smi.split('.'))

        pred_text = os.path.join(pred_text_dir, f'{idx}.txt')

        with torch.no_grad():
            top_k_results = beam_model.run_search(prod_smi=p, max_steps=args.max_steps, rxn_class=rxn_class)

        beam_matched = False
        valid_pred_count = 0
        with open(pred_text, 'w') as fp:
            for beam_idx, path in enumerate(top_k_results):
                pred_smi = path['final_smi']
                if pred_smi != 'final_smi_unmapped':
                    pred_result = smi_tokenizer(pred_smi)
                    fp.write(f'{pred_result}\n')
                    valid_pred_count += 1
                pred_set = set(pred_smi.split('.'))
                if pred_set == r_set and not beam_matched:
                    top_k[beam_idx] += 1
                    beam_matched = True
                    true_idx = beam_idx

        all_predictions = []
        if not args.no_round_trip and valid_pred_count > 0:
            opt = build_round_trip_options(pred_text, args)
            init_logger(opt.log_file)
            translator = build_translator(opt, report_score=True)
            _, all_predictions = translator.translate(src_path=opt.src,
                                                       tgt_path=opt.tgt,
                                                       src_dir=opt.src_dir,
                                                       batch_size=opt.batch_size,
                                                       attn_debug=opt.attn_debug)
        
        p_mol = Chem.MolFromSmiles(p)
        [a.ClearProp('molAtomMapNumber') for a in p_mol.GetAtoms()]
        p_mol = Chem.MolFromSmiles(Chem.MolToSmiles(p_mol))
        p_smi = Chem.MolToSmiles(p_mol)

        if not args.no_round_trip:
            rt_matched = False
            for rt_idx, predictions in enumerate(all_predictions):
                for prediction in predictions:
                    pred = (''.join(prediction.strip().split(' ')))
                    mol = Chem.MolFromSmiles(pred)
                    if mol is not None:
                        pred = Chem.MolToSmiles(mol, isomericSmiles=True)
                    else:
                        pred = ''

                    if beam_matched and not rt_matched:
                        if true_idx <= rt_idx:
                            rt_top_k[true_idx] += 1
                            rt_matched = True
                        elif pred == p_smi:
                            rt_top_k[rt_idx] += 1
                            rt_matched = True
                    if pred == p_smi and not beam_matched and not rt_matched:
                        rt_top_k[rt_idx] += 1
                        rt_matched = True


        msg = ''
        for beam_idx in [1, 3, 5, 10, 20, 50]:
            match_acc = np.sum(top_k[:beam_idx]) / (idx + 1)
            msg += 'Exact accuracy, t%d: %.3f' % (beam_idx, match_acc)
            if not args.no_round_trip:
                Rt_acc = np.sum(rt_top_k[:beam_idx]) / (idx + 1)
                msg += ' Round-trip accuracy, t%d: %.3f ' % (beam_idx, Rt_acc)
        p_bar.set_description(msg)


if __name__ == '__main__':
    main()
