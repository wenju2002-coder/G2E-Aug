from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare_data import apply_edit_to_mol
from rdkit import Chem
from utils.class_balance import effective_num_weights, posterior_logit_adjustment
from utils.collate_fn import get_batch_graphs
from utils.rxn_graphs import MolGraph, Vocab

from models.encoder import Global_Attention, MPNEncoder
from models.model_utils import (creat_edits_feats, index_select_ND,
                                unbatch_feats)


class Graph2Edits(nn.Module):
    def __init__(self,
                 config: Dict,
                 atom_vocab: Vocab,
                 bond_vocab: Vocab,
                 device: str = 'cpu') -> None:
        """
        Parameters
        ----------
        config: Dict, Model arguments
        atom_vocab: atom and LG edit labels
        bond_vocab: bond edit labels
        device: str, Device to run the model on.
        """
        super(Graph2Edits, self).__init__()

        self.config = config
        self.atom_vocab = atom_vocab
        self.bond_vocab = bond_vocab
        self.atom_outdim = len(atom_vocab)
        self.bond_outdim = len(bond_vocab)
        self.device = device

        self._build_layers()

    def _build_layers(self) -> None:
        """Builds the different layers associated with the model."""
        config = self.config
        self.encoder = MPNEncoder(atom_fdim=config['n_atom_feat'],
                                  bond_fdim=config['n_bond_feat'],
                                  hidden_size=config['mpn_size'],
                                  depth=config['depth'],
                                  dropout=config['dropout_mpn'],
                                  atom_message=config['atom_message'])

        self.W_vv = nn.Linear(config['mpn_size'],
                              config['mpn_size'], bias=False)
        nn.init.eye_(self.W_vv.weight)
        self.W_vc = nn.Linear(config['mpn_size'],
                              config['mpn_size'], bias=False)

        if config['use_attn']:
            self.attn = Global_Attention(
                d_model=config['mpn_size'], heads=config['n_heads'])

        self.atom_linear = nn.Sequential(
            nn.Linear(config['mpn_size'], config['mlp_size']),
            nn.ReLU(),
            nn.Dropout(p=config['dropout_mlp']),
            nn.Linear(config['mlp_size'], self.atom_outdim))

        lg_gcn_dim = config.get('lg_gcn_dim') or min(config['mlp_size'], max(64, self.atom_outdim))
        self.lg_label_embedding = nn.Embedding(self.atom_outdim, lg_gcn_dim)
        self.lg_gcn_in = nn.Linear(lg_gcn_dim, lg_gcn_dim)
        self.lg_gcn_out = nn.Linear(lg_gcn_dim, lg_gcn_dim)
        self.lg_gcn_norm = nn.LayerNorm(lg_gcn_dim)
        self.lg_label_bias = nn.Linear(lg_gcn_dim, 1)
        self.register_buffer('lg_cooccurrence_adj', torch.eye(self.atom_outdim))
        self.register_buffer('lg_label_mask', torch.tensor([
            1.0 if isinstance(elem, tuple) and len(elem) > 0 and elem[0] == 'Attaching LG' else 0.0
            for elem in self.atom_vocab.elem_list
        ], dtype=torch.float))
        self.register_buffer('bond_logit_adjustment', torch.zeros(self.bond_outdim))
        self.register_buffer('atom_logit_adjustment', torch.zeros(self.atom_outdim))
        self.register_buffer('terminate_logit_adjustment', torch.zeros(1))
        self.register_buffer('bond_class_weights', torch.ones(self.bond_outdim))
        self.register_buffer('atom_class_weights', torch.ones(self.atom_outdim))
        self.register_buffer('terminate_class_weight', torch.ones(1))
        self.register_buffer('atom_class_counts', torch.zeros(self.atom_outdim))
        self.register_buffer('lg_head_mask', torch.zeros(self.atom_outdim))
        self.register_buffer('lg_tail_mask', torch.zeros(self.atom_outdim))
        self.register_buffer('lg_tail_to_head_adj', torch.zeros(self.atom_outdim, self.atom_outdim))
        self.bond_linear = nn.Sequential(
            nn.Linear(config['mpn_size'] * 2, config['mlp_size']),
            nn.ReLU(),
            nn.Dropout(p=config['dropout_mlp']),
            nn.Linear(config['mlp_size'], self.bond_outdim))

        self.bond_break_linear = nn.Sequential(
            nn.Linear(config['mpn_size'] * 2 + 1, config['mlp_size']),
            nn.SELU(),
            nn.Dropout(p=config['dropout_mlp']),
            nn.Linear(config['mlp_size'], 1))

        self.graph_linear = nn.Sequential(
            nn.Linear(config['mpn_size'], config['mlp_size']),
            nn.ReLU(),
            nn.Dropout(p=config['dropout_mlp']),
            nn.Linear(config['mlp_size'], 1))

    def set_lg_cooccurrence_adj(self, adj: torch.Tensor) -> None:
        """Sets the normalized LG co-occurrence graph used by the label GCN."""
        if adj.size(0) != self.atom_outdim or adj.size(1) != self.atom_outdim:
            raise ValueError(f'LG adjacency shape {tuple(adj.size())} does not match atom vocab size {self.atom_outdim}')
        self.lg_cooccurrence_adj = adj.to(self.device).float()
        if self.atom_class_counts.sum().item() > 0:
            self._refresh_lg_neighbor_masks()

    def set_edit_class_statistics(self, stats: Dict, initialize_lg_tail: bool = False) -> None:
        bond_counts = stats.get('bond_counts', torch.ones(self.bond_outdim))
        atom_counts = stats.get('atom_counts', torch.ones(self.atom_outdim))
        terminate_count = stats.get('terminate_count', torch.tensor(1.0))

        bond_counts = torch.as_tensor(bond_counts, dtype=torch.float)
        atom_counts = torch.as_tensor(atom_counts, dtype=torch.float)
        terminate_count = torch.as_tensor(terminate_count, dtype=torch.float).reshape(1)
        if bond_counts.numel() != self.bond_outdim:
            bond_counts = torch.ones(self.bond_outdim, dtype=torch.float)
        if atom_counts.numel() != self.atom_outdim:
            atom_counts = torch.ones(self.atom_outdim, dtype=torch.float)

        full_counts = torch.cat([bond_counts, atom_counts, terminate_count], dim=0)
        prior_smoothing = self.config.get('class_prior_smoothing', 1.0)
        edit_tau = self.config.get('logit_adjust_tau', 0.0)
        full_adjustment = posterior_logit_adjustment(
            full_counts, tau=edit_tau, smoothing=prior_smoothing)

        bond_adjustment = full_adjustment[:self.bond_outdim]
        atom_adjustment = full_adjustment[self.bond_outdim:self.bond_outdim + self.atom_outdim]
        terminate_adjustment = full_adjustment[-1:]

        lg_mask = self.lg_label_mask.detach().cpu().bool()
        lg_counts = atom_counts * lg_mask.float()
        lg_tau = self.config.get('lg_logit_adjust_tau', edit_tau)
        if lg_tau > 0 and lg_mask.any() and lg_counts[lg_mask].sum().item() > 0:
            lg_adjustment = posterior_logit_adjustment(
                lg_counts[lg_mask], tau=lg_tau, smoothing=prior_smoothing)
            atom_adjustment = atom_adjustment.clone()
            atom_adjustment[lg_mask] = lg_adjustment

        beta = self.config.get('class_balanced_beta', 0.9999)
        full_weights = effective_num_weights(full_counts, beta=beta)
        bond_weights = full_weights[:self.bond_outdim]
        atom_weights = full_weights[self.bond_outdim:self.bond_outdim + self.atom_outdim]
        terminate_weight = full_weights[-1:]

        lg_beta = self.config.get('lg_class_balanced_beta', beta)
        if lg_mask.any() and lg_counts[lg_mask].sum().item() > 0:
            lg_weights = effective_num_weights(lg_counts[lg_mask], beta=lg_beta)
            atom_weights = atom_weights.clone()
            atom_weights[lg_mask] = lg_weights

        self.bond_logit_adjustment = bond_adjustment.to(self.device)
        self.atom_logit_adjustment = atom_adjustment.to(self.device)
        self.terminate_logit_adjustment = terminate_adjustment.to(self.device)
        self.bond_class_weights = bond_weights.to(self.device)
        self.atom_class_weights = atom_weights.to(self.device)
        self.terminate_class_weight = terminate_weight.to(self.device)
        self.atom_class_counts = atom_counts.to(self.device)
        self._refresh_lg_neighbor_masks()

        if initialize_lg_tail:
            self.initialize_tail_lg_embeddings()

    def _refresh_lg_neighbor_masks(self) -> None:
        counts = self.atom_class_counts.to(self.device).float()
        lg_mask = self.lg_label_mask.to(self.device).bool()
        positive_lg = lg_mask & (counts > 0)
        head = positive_lg & (counts >= self.config.get('lg_head_threshold', 50.0))
        tail = positive_lg & (counts <= self.config.get('lg_tail_threshold', 10.0))

        if not head.any() and positive_lg.any():
            positive_idx = torch.where(positive_lg)[0]
            k = max(1, int(0.2 * positive_idx.numel()))
            _, top_pos = torch.topk(counts[positive_idx], k=k)
            head = torch.zeros_like(positive_lg)
            head[positive_idx[top_pos]] = True

        if not tail.any() and positive_lg.any():
            tail = positive_lg & ~head

        adj = self.lg_cooccurrence_adj.to(self.device).float().clamp_min(0.0)
        tail_to_head = adj * tail.float().unsqueeze(1) * head.float().unsqueeze(0)
        row_sum = tail_to_head.sum(dim=1, keepdim=True)

        empty_tail = tail & (row_sum.squeeze(1) <= 0)
        if empty_tail.any() and head.any():
            head_prior = counts * head.float()
            if head_prior.sum().item() <= 0:
                head_prior = head.float()
            head_prior = head_prior / head_prior.sum().clamp_min(1e-12)
            tail_to_head[empty_tail] = head_prior.unsqueeze(0).expand(
                int(empty_tail.sum().item()), -1)
            row_sum = tail_to_head.sum(dim=1, keepdim=True)

        tail_to_head = tail_to_head / row_sum.clamp_min(1e-12)
        tail_to_head = tail_to_head * tail.float().unsqueeze(1)

        self.lg_head_mask = head.float()
        self.lg_tail_mask = tail.float()
        self.lg_tail_to_head_adj = tail_to_head

    def initialize_tail_lg_embeddings(self) -> None:
        alpha = self.config.get('lg_neighbor_init_alpha', 0.0)
        if alpha <= 0:
            return

        tail = self.lg_tail_mask.to(self.device).bool()
        if not tail.any():
            return

        with torch.no_grad():
            prototypes = torch.mm(self.lg_tail_to_head_adj.to(self.device),
                                  self.lg_label_embedding.weight.detach())
            self.lg_label_embedding.weight[tail] = (
                (1.0 - alpha) * self.lg_label_embedding.weight[tail] +
                alpha * prototypes[tail])

    def compute_lg_neighbor_regularization(self) -> torch.Tensor:
        tail = self.lg_tail_mask.to(self.device).bool()
        if not tail.any():
            return torch.tensor(0.0, device=self.device)

        prototypes = torch.mm(self.lg_tail_to_head_adj.to(self.device),
                              self.lg_label_embedding.weight).detach()
        return F.mse_loss(self.lg_label_embedding.weight[tail], prototypes[tail])

    def get_edit_target_weight(self, target_idx: torch.Tensor, n_bonds: int, flat_dim: int) -> torch.Tensor:
        if not self.config.get('use_class_balanced_loss', False):
            return torch.tensor(1.0, device=self.device)

        target = int(target_idx.item())
        max_bond_idx = n_bonds * self.bond_outdim
        if target < max_bond_idx:
            return self.bond_class_weights[target % self.bond_outdim]
        if target == flat_dim - 1:
            return self.terminate_class_weight.squeeze(0)
        return self.atom_class_weights[(target - max_bond_idx) % self.atom_outdim]

    def compute_lg_logit_bias(self) -> torch.Tensor:
        adj = self.lg_cooccurrence_adj.to(self.device)
        label_feats = self.lg_label_embedding.weight
        hidden = torch.mm(adj, label_feats)
        hidden = F.relu(self.lg_gcn_in(hidden))
        hidden = F.dropout(hidden,
                           p=self.config.get('lg_gcn_dropout', self.config['dropout_mlp']),
                           training=self.training)
        hidden = torch.mm(adj, hidden)
        hidden = self.lg_gcn_out(hidden)
        hidden = self.lg_gcn_norm(hidden + label_feats)
        return self.lg_label_bias(torch.tanh(hidden)).squeeze(-1) * self.lg_label_mask.to(self.device)

    def apply_logit_adjustment(self,
                               bond_outs: torch.Tensor,
                               atom_outs: torch.Tensor,
                               graph_outs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.config.get('use_logit_adjustment', True):
            return bond_outs, atom_outs, graph_outs
        if self.training and not self.config.get('logit_adjust_train', False):
            return bond_outs, atom_outs, graph_outs

        bond_outs = bond_outs + self.bond_logit_adjustment.to(self.device).unsqueeze(0)
        atom_outs = atom_outs + self.atom_logit_adjustment.to(self.device).unsqueeze(0)
        graph_outs = graph_outs + self.terminate_logit_adjustment.to(self.device).view(1, 1)
        return bond_outs, atom_outs, graph_outs

    def expected_f_bond_dim(self) -> int:
        if self.config.get('atom_message', False):
            return self.config['n_atom_feat'] + self.config['n_bond_feat']
        return self.config['n_bond_feat']

    def to_device(self, tensors: Union[List, torch.Tensor]) -> Union[List, torch.Tensor]:
        """Converts all inputs to the device used.

        Parameters
        ----------
        tensors: Union[List, torch.Tensor],
            Tensors to convert to model device. The tensors can be either a
            single tensor or an iterable of tensors.
        """
        if isinstance(tensors, list) or isinstance(tensors, tuple):
            tensors = [tensor.to(self.device, non_blocking=True)
                       for tensor in tensors]
            return tensors
        elif isinstance(tensors, torch.Tensor):
            return tensors.to(self.device, non_blocking=True)
        else:
            raise ValueError(f"Tensors of type {type(tensors)} unsupported")

    def compute_edit_scores(self, prod_tensors: Tuple[torch.Tensor],
                            prod_scopes: Tuple[List], prev_atom_hiddens: torch.Tensor = None,
                            prev_atom_scope: Tuple[List] = None) -> Tuple[torch.Tensor]:
        """Computes the edit scores given product tensors and scopes.

        Parameters
        ----------
        prod_tensors: Tuple[torch.Tensor]:
            Product tensors
        prod_scopes: Tuple[List]
            Product scopes. Scopes is composed of atom and bond scopes, which
            keep track of atom and bond indices for each molecule in the 2D
            feature list
        prev_atom_hiddens: torch.Tensor, default None,
            Previous hidden state of atoms.
        """
        prod_tensors = self.to_device(prod_tensors)
        expected_f_bond_dim = self.expected_f_bond_dim()
        if prod_tensors[1].size(1) != expected_f_bond_dim:
            raise ValueError(
                f'Bond feature dimension {prod_tensors[1].size(1)} does not match model config '
                f'{expected_f_bond_dim}. Re-run prepare_data.py after enabling AEE features.')
        atom_scope, bond_scope = prod_scopes
        if prev_atom_hiddens is None:
            n_atoms = prod_tensors[0].size(0)
            prev_atom_hiddens = torch.zeros(
                n_atoms, self.config['mpn_size'], device=self.device)

        a_feats = self.encoder(prod_tensors, mask=None)
        if self.config['use_attn']:
            feats, mask = creat_edits_feats(a_feats, atom_scope)
            attention_score, feats = self.attn(feats, mask)
            a_feats = unbatch_feats(feats, atom_scope)

        if a_feats.shape[0] != prev_atom_hiddens.shape[0]:
            n_atoms = a_feats.shape[0]
            new_ha = torch.zeros(
                n_atoms, self.config['mpn_size'], device=self.device)
            for idx, ((st_n, le_n), (st_p, le_p)) in enumerate(zip(*(atom_scope, prev_atom_scope))):
                new_ha[st_n: st_n + le_p] = prev_atom_hiddens[st_p: st_p + le_p]
            prev_atom_hiddens = new_ha

        assert a_feats.shape == prev_atom_hiddens.shape
        atom_feats = F.relu(self.W_vv(prev_atom_hiddens) + self.W_vc(a_feats))
        prev_atom_hiddens = atom_feats.clone()
        prev_atom_scope = atom_scope

        node_feats = atom_feats.clone()

        undirected_b2a = prod_tensors[7]          # 正确的 undirected bond to atom mapping
        bond_energy = prod_tensors[8] if len(prod_tensors) > 8 else torch.zeros(undirected_b2a.size(0), 1, device=self.device)

        bond_starts = index_select_ND(atom_feats, index=undirected_b2a[:, 0])
        bond_ends = index_select_ND(atom_feats, index=undirected_b2a[:, 1])

        bond_feats = torch.cat([bond_starts, bond_ends], dim=1)
        bond_break_feats = torch.cat([bond_feats, bond_energy], dim=1)
        graph_vecs = torch.stack(
            [atom_feats[st: st + le].sum(dim=0) for st, le in atom_scope])

        atom_outs = self.atom_linear(node_feats) + self.compute_lg_logit_bias().unsqueeze(0)
        bond_outs = self.bond_linear(bond_feats)
        bond_break_outs = self.bond_break_linear(bond_break_feats).squeeze(-1)
        graph_outs = self.graph_linear(graph_vecs)
        bond_outs, atom_outs, graph_outs = self.apply_logit_adjustment(
            bond_outs, atom_outs, graph_outs)

        edit_scores = [torch.cat([bond_outs[st_b: st_b + le_b].flatten(),
                                  atom_outs[st_a: st_a + le_a].flatten(), graph_outs[idx]], dim=-1)
                       for idx, ((st_a, le_a), (st_b, le_b)) in enumerate(zip(*(atom_scope, bond_scope)))]
        bond_break_scores = [bond_break_outs[st_b: st_b + le_b]
                             for st_b, le_b in bond_scope]
        bond_energy_scores = [bond_energy[st_b: st_b + le_b].squeeze(-1)
                              for st_b, le_b in bond_scope]
        return edit_scores, prev_atom_hiddens, prev_atom_scope, graph_vecs, bond_break_scores, bond_energy_scores

    def compute_graph_vecs(self, graph_inputs: Tuple[torch.Tensor, List]) -> torch.Tensor:
        graph_tensors, graph_scopes = graph_inputs
        graph_tensors = self.to_device(graph_tensors)
        expected_f_bond_dim = self.expected_f_bond_dim()
        if graph_tensors[1].size(1) != expected_f_bond_dim:
            raise ValueError(
                f'Bond feature dimension {graph_tensors[1].size(1)} does not match model config '
                f'{expected_f_bond_dim}. Re-run prepare_data.py after enabling AEE features.')
        atom_scope, _ = graph_scopes
        atom_feats = self.encoder(graph_tensors, mask=None)
        if self.config['use_attn']:
            feats, mask = creat_edits_feats(atom_feats, atom_scope)
            _, feats = self.attn(feats, mask)
            atom_feats = unbatch_feats(feats, atom_scope)
        return torch.stack([atom_feats[st: st + le].sum(dim=0) for st, le in atom_scope])

    def forward(self, prod_seq_inputs: List[Tuple[torch.Tensor, List]]) -> Tuple[torch.Tensor]:
        """
        Forward propagation step.

        Parameters
        ----------
        prod_seq_inputs: List[Tuple[torch.Tensor, List]]
            List of prod_tensors for edit sequence
        """
        max_seq_len = len(prod_seq_inputs)
        assert len(prod_seq_inputs[0]) == 2

        prev_atom_hiddens = None
        prev_atom_scope = None
        seq_edit_scores = []
        seq_bond_break_scores = []
        seq_bond_energy = []
        seq_graph_vecs = []
        for idx in range(max_seq_len):
            prod_tensors, prod_scopes = prod_seq_inputs[idx]
            edit_scores, prev_atom_hiddens, prev_atom_scope, graph_outs, bond_break_scores, bond_energy = self.compute_edit_scores(
                prod_tensors, prod_scopes, prev_atom_hiddens, prev_atom_scope)
            seq_edit_scores.append(edit_scores)
            seq_bond_break_scores.append(bond_break_scores)
            seq_bond_energy.append(bond_energy)
            seq_graph_vecs.append(graph_outs)
        return seq_edit_scores, seq_bond_break_scores, seq_bond_energy, seq_graph_vecs

    def predict(self, prod_smi: str, rxn_class: int = None, max_steps: int = 9):
        """Make predictions for given product smiles string.

        Parameters
        ----------
        prod_smi: str,
            Product SMILES string
        rxn_class: int, default None
            Associated reaction class for the product
        max_steps: int, default 8
            Max number of edit steps allowed
        """
        use_rxn_class = False
        if rxn_class is not None:
            use_rxn_class = True

        done = False
        steps = 0
        edits = []
        edits_atom = []
        prev_atom_hiddens = None
        prev_atom_scope = None

        products = Chem.MolFromSmiles(prod_smi)
        Chem.Kekulize(products)
        prod_graph = MolGraph(mol=Chem.Mol(products),
                              rxn_class=rxn_class, use_rxn_class=use_rxn_class)
        prod_tensors, prod_scopes = get_batch_graphs(
            [prod_graph], use_rxn_class=use_rxn_class)

        while not done and steps <= max_steps:
            if prod_tensors[-2].size() == (1, 0):
                edit = 'Terminate'
                edits.append(edit)
                done = True
                break

            edit_logits, prev_atom_hiddens, prev_atom_scope, graph_outs, _, _ = self.compute_edit_scores(
                prod_tensors, prod_scopes, prev_atom_hiddens, prev_atom_scope)
            idx = torch.argmax(edit_logits[0])
            val = edit_logits[0][idx]

            max_bond_idx = products.GetNumBonds() * self.bond_outdim

            if idx.item() == len(edit_logits[0]) - 1:
                edit = 'Terminate'
                edits.append(edit)
                done = True
                break

            elif idx.item() < max_bond_idx:
                bond_logits = edit_logits[0][:products.GetNumBonds(
                ) * self.bond_outdim]
                bond_logits = bond_logits.reshape(
                    products.GetNumBonds(), self.bond_outdim)
                idx_tensor = torch.where(bond_logits == val)

                idx_tensor = [indices[-1] for indices in idx_tensor]

                bond_idx, edit_idx = idx_tensor[0].item(), idx_tensor[1].item()
                a1 = products.GetBondWithIdx(
                    bond_idx).GetBeginAtom().GetAtomMapNum()
                a2 = products.GetBondWithIdx(
                    bond_idx).GetEndAtom().GetAtomMapNum()

                a1, a2 = sorted([a1, a2])
                edit_atom = [a1, a2]
                edit = self.bond_vocab.get_elem(edit_idx)

            else:
                atom_logits = edit_logits[0][max_bond_idx:-1]

                assert len(atom_logits) == products.GetNumAtoms() * \
                    self.atom_outdim
                atom_logits = atom_logits.reshape(
                    products.GetNumAtoms(), self.atom_outdim)
                idx_tensor = torch.where(atom_logits == val)

                idx_tensor = [indices[-1] for indices in idx_tensor]
                atom_idx, edit_idx = idx_tensor[0].item(), idx_tensor[1].item()

                a1 = products.GetAtomWithIdx(atom_idx).GetAtomMapNum()
                edit_atom = a1
                edit = self.atom_vocab.get_elem(edit_idx)

            try:
                products = apply_edit_to_mol(mol=Chem.Mol(
                    products), edit=edit, edit_atom=edit_atom)
                prod_graph = MolGraph(mol=Chem.Mol(
                    products),  rxn_class=rxn_class, use_rxn_class=use_rxn_class)
                prod_tensors, prod_scopes = get_batch_graphs(
                    [prod_graph], use_rxn_class=use_rxn_class)

                edits.append(edit)
                edits_atom.append(edit_atom)
                steps += 1

            except:
                steps += 1
                continue

        return edits, edits_atom

    def get_saveables(self) -> Dict:
        """
        Return the attributes of model used for its construction. This is used
        in restoring the model.
        """
        saveables = {}
        saveables['config'] = self.config
        saveables['atom_vocab'] = self.atom_vocab
        saveables['bond_vocab'] = self.bond_vocab

        return saveables
