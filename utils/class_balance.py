import os
from collections import Counter
from typing import Dict, Iterable, Union

import joblib
import torch


def _torch_load(path: str):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _as_elem_list(vocab_or_list: Union[Iterable, object]):
    if hasattr(vocab_or_list, 'elem_list'):
        return list(vocab_or_list.elem_list)
    return list(vocab_or_list)


def build_edit_class_stats(rxns_data, bond_vocab, atom_vocab) -> Dict[str, torch.Tensor]:
    bond_labels = _as_elem_list(bond_vocab)
    atom_labels = _as_elem_list(atom_vocab)
    bond_to_idx = {label: idx for idx, label in enumerate(bond_labels)}
    atom_to_idx = {label: idx for idx, label in enumerate(atom_labels)}

    bond_counts = torch.zeros(len(bond_labels), dtype=torch.float)
    atom_counts = torch.zeros(len(atom_labels), dtype=torch.float)
    terminate_count = torch.tensor(0.0, dtype=torch.float)

    action_counter = Counter()
    for rxn in rxns_data:
        for edit in rxn.edits:
            action_counter[edit] += 1
            if edit == 'Terminate':
                terminate_count += 1.0
            elif edit in bond_to_idx:
                bond_counts[bond_to_idx[edit]] += 1.0
            elif edit in atom_to_idx:
                atom_counts[atom_to_idx[edit]] += 1.0

    return {
        'bond_counts': bond_counts,
        'atom_counts': atom_counts,
        'terminate_count': terminate_count,
        'action_counter': action_counter,
    }


def load_or_build_edit_class_stats(data_root: str, bond_vocab, atom_vocab) -> Dict[str, torch.Tensor]:
    stats_file = os.path.join(data_root, 'train', 'edit_class_stats.pt')
    if os.path.exists(stats_file):
        return _torch_load(stats_file)

    train_file = os.path.join(data_root, 'train', 'train.file.kekulized')
    if not os.path.exists(train_file):
        train_file = os.path.join(data_root, 'train', 'train.file')
    if not os.path.exists(train_file):
        return {
            'bond_counts': torch.ones(len(bond_vocab), dtype=torch.float),
            'atom_counts': torch.ones(len(atom_vocab), dtype=torch.float),
            'terminate_count': torch.tensor(1.0, dtype=torch.float),
            'action_counter': Counter(),
        }

    rxns_data = joblib.load(train_file)
    stats = build_edit_class_stats(rxns_data, bond_vocab, atom_vocab)
    torch.save(stats, stats_file)

    action_freq_file = os.path.join(data_root, 'train', 'action_freq_map.pkl')
    if not os.path.exists(action_freq_file):
        joblib.dump(stats['action_counter'], action_freq_file)

    return stats


def effective_num_weights(counts: torch.Tensor, beta: float = 0.9999) -> torch.Tensor:
    counts = counts.float()
    weights = torch.ones_like(counts)
    positive = counts > 0
    if positive.any():
        if beta <= 0:
            weights[positive] = 1.0
        elif beta >= 1:
            weights[positive] = 1.0 / counts[positive].clamp_min(1.0)
        else:
            weights[positive] = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta, device=counts.device), counts[positive]))

        weights[~positive] = 0.0
        weights[positive] = weights[positive] / weights[positive].mean().clamp_min(1e-12)

    return weights


def posterior_logit_adjustment(counts: torch.Tensor,
                               tau: float = 1.0,
                               smoothing: float = 1.0) -> torch.Tensor:
    if tau <= 0:
        return torch.zeros_like(counts, dtype=torch.float)

    counts = counts.float() + smoothing
    prior = counts / counts.sum().clamp_min(1e-12)
    return -tau * torch.log(prior.clamp_min(1e-12))


def load_lg_cooccurrence_adj(path: str, atom_vocab_size: int) -> torch.Tensor:
    if not os.path.exists(path):
        return torch.eye(atom_vocab_size, dtype=torch.float)

    artifact = _torch_load(path)
    if isinstance(artifact, dict):
        artifact = artifact.get('adj', artifact.get('lg_cooccurrence_adj'))
    if artifact is None:
        return torch.eye(atom_vocab_size, dtype=torch.float)

    if artifact.size(0) != atom_vocab_size or artifact.size(1) != atom_vocab_size:
        return torch.eye(atom_vocab_size, dtype=torch.float)

    return artifact.float()
