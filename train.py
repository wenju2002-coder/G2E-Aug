import argparse
import os
import sys
import joblib
from datetime import datetime as dt

import torch
import torch.nn as nn
from rdkit import RDLogger
from torch.optim import Adam, lr_scheduler

from models import Graph2Edits
from models.model_utils import CSVLogger, get_seq_edit_accuracy
from utils.class_balance import load_lg_cooccurrence_adj, load_or_build_edit_class_stats
from utils.datasets import RetroEditDataset, RetroEvalDataset
from utils.mol_features import ATOM_FDIM, BOND_FDIM
from utils.rxn_graphs import Vocab

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

DATE_TIME = dt.now().strftime('%d-%m-%Y--%H-%M-%S')
ROOT_DIR = './'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_delete_bond_indices(bond_vocab):
    return [idx for idx, elem in bond_vocab.idx_to_elem.items()
            if isinstance(elem, tuple) and len(elem) > 0 and elem[0] == 'Delete Bond']


def derive_bond_break_labels(edit_labels, bond_scores, bond_vocab_size, delete_bond_indices=None):
    bond_break_labels = []
    for label, score in zip(edit_labels, bond_scores):
        n_bonds = score.size(0)
        bond_label = label[:n_bonds * bond_vocab_size].reshape(n_bonds, bond_vocab_size)
        if delete_bond_indices:
            bond_break_labels.append((bond_label[:, delete_bond_indices].sum(dim=1) > 0).float())
        else:
            bond_break_labels.append((bond_label.sum(dim=1) > 0).float())
    return bond_break_labels


def compute_bond_break_loss(seq_bond_scores, seq_bond_labels, seq_bond_energy, seq_labels, seq_mask, model, bce_loss,
                            delete_bond_indices=None):
    max_seq_len, batch_size = seq_mask.size()
    bond_losses = []

    for idx in range(max_seq_len):
        if seq_bond_labels is None:
            labels_idx = derive_bond_break_labels(
                model.to_device(seq_labels[idx]), seq_bond_scores[idx], model.bond_outdim, delete_bond_indices)
        else:
            labels_idx = model.to_device(seq_bond_labels[idx])

        for batch_idx in range(batch_size):
            if seq_mask[idx][batch_idx].item() == 0:
                continue

            logits = seq_bond_scores[idx][batch_idx]
            labels = labels_idx[batch_idx].to(logits.device).float()
            if logits.numel() == 0:
                continue

            energy = seq_bond_energy[idx][batch_idx].to(logits.device).float()
            weights = 1.0 + model.config.get('bond_energy_loss_alpha', 1.0) * (1.0 - energy)
            loss = bce_loss(logits, labels) * weights
            bond_losses.append(loss.mean())

    if not bond_losses:
        return torch.tensor(0.0, device=seq_mask.device)

    return torch.stack(bond_losses).mean()


def compute_graph_alignment_loss(seq_graph_vecs, reac_seq_tensors, seq_mask, model, mse_loss):
    if reac_seq_tensors is None:
        return torch.tensor(0.0, device=seq_mask.device)

    max_seq_len, batch_size = seq_mask.size()
    align_losses = []
    for idx in range(max_seq_len):
        prod_vecs = seq_graph_vecs[idx]
        reac_vecs = model.compute_graph_vecs(reac_seq_tensors[idx])
        for batch_idx in range(batch_size):
            if seq_mask[idx][batch_idx].item() == 0:
                continue
            align_losses.append(mse_loss(prod_vecs[batch_idx], reac_vecs[batch_idx]))

    if not align_losses:
        return torch.tensor(0.0, device=seq_mask.device)

    return torch.stack(align_losses).mean()


def build_model_config(args):
    model_config = {}
    if args.get('use_rxn_class', False):
        atom_fdim = ATOM_FDIM + 10
    else:
        atom_fdim = ATOM_FDIM
    model_config['n_atom_feat'] = atom_fdim
    if args.get('atom_message', False):
        model_config['n_bond_feat'] = BOND_FDIM
    else:
        model_config['n_bond_feat'] = atom_fdim + BOND_FDIM
    model_config['mpn_size'] = args['mpn_size']
    model_config['mlp_size'] = args['mlp_size']
    model_config['depth'] = args['depth']
    model_config['dropout_mlp'] = args['dropout_mlp']
    model_config['dropout_mpn'] = args['dropout_mpn']
    model_config['atom_message'] = args['atom_message']
    model_config['use_attn'] = args['use_attn']
    model_config['n_heads'] = args['n_heads']
    model_config['bond_energy_loss_alpha'] = args['bond_energy_loss_alpha']
    model_config['use_logit_adjustment'] = not args.get('disable_logit_adjustment', False)
    model_config['logit_adjust_train'] = args.get('logit_adjust_train', False)
    model_config['logit_adjust_tau'] = args['logit_adjust_tau']
    model_config['lg_logit_adjust_tau'] = args['lg_logit_adjust_tau']
    model_config['class_prior_smoothing'] = args['class_prior_smoothing']
    model_config['use_class_balanced_loss'] = not args.get('disable_class_balanced_loss', False)
    model_config['class_balanced_beta'] = args['class_balanced_beta']
    model_config['lg_class_balanced_beta'] = args['lg_class_balanced_beta']
    model_config['lg_gcn_dim'] = args['lg_gcn_dim']
    model_config['lg_gcn_dropout'] = args['lg_gcn_dropout']
    model_config['lg_head_threshold'] = args['lg_head_threshold']
    model_config['lg_tail_threshold'] = args['lg_tail_threshold']
    model_config['lg_neighbor_init_alpha'] = args['lg_neighbor_init_alpha']

    return model_config


def save_checkpoint(model, path, epoch):
    save_dict = {'state': model.state_dict()}
    if hasattr(model, 'get_saveables'):
        save_dict['saveables'] = model.get_saveables()

    name = f'epoch_{epoch + 1}.pt'
    save_file = os.path.join(path, name)
    torch.save(save_dict, save_file)

def schedule_class_weight(raw_weight, args, epoch):
    if args.get('disable_class_balanced_loss', False):
        return raw_weight

    weight = raw_weight
    min_weight = args.get('class_weight_min', 0.2)
    max_weight = args.get('class_weight_max', 5.0)
    weight = torch.clamp(weight, min=min_weight, max=max_weight)

    warmup_epochs = args.get('class_balanced_warmup_epochs', 0)
    if warmup_epochs > 0:
        progress = min(1.0, float(epoch + 1) / float(warmup_epochs))
        weight = 1.0 + progress * (weight - 1.0)

    return weight
def train_epoch(args, epoch, model, train_data, loss_fn, bce_loss, mse_loss, optimizer):
    torch.cuda.empty_cache()
    model.train()
    train_loss = 0
    train_acc = 0
    for batch_id, batch_data in enumerate(train_data):
        graph_seq_tensors, seq_labels, seq_mask, seq_bond_labels, _, reac_seq_tensors = batch_data
        seq_mask = seq_mask.to(DEVICE)
        seq_edit_scores, seq_bond_scores, seq_bond_energy, seq_graph_vecs = model(graph_seq_tensors)

        max_seq_len, batch_size = seq_mask.size()
        seq_loss = []

        for idx in range(max_seq_len):
            edit_labels_idx = model.to_device(seq_labels[idx])
            loss_batch = []
            for i in range(batch_size):
                target = torch.argmax(edit_labels_idx[i]).unsqueeze(0).long()
                sample_loss = loss_fn(seq_edit_scores[idx][i].unsqueeze(0), target).sum()
                target_weight = model.get_edit_target_weight(
                    target.squeeze(0), seq_bond_scores[idx][i].size(0), seq_edit_scores[idx][i].size(0))
                target_weight = schedule_class_weight(target_weight, args, epoch)

                loss_batch.append(seq_mask[idx][i] * sample_loss * target_weight)

            loss = torch.stack(loss_batch, dim=0).mean()
            seq_loss.append(loss)
        bond_break_loss = compute_bond_break_loss(
            seq_bond_scores, seq_bond_labels, seq_bond_energy, seq_labels, seq_mask,
            model, bce_loss, args.get('delete_bond_indices'))
        graph_alignment_loss = compute_graph_alignment_loss(
            seq_graph_vecs, reac_seq_tensors, seq_mask, model, mse_loss)
        lg_neighbor_reg_loss = model.compute_lg_neighbor_regularization()
        total_loss = torch.stack(seq_loss).mean() + \
            args['bond_break_loss_weight'] * bond_break_loss + \
            args['graph_alignment_loss_weight'] * graph_alignment_loss + \
            args['lg_neighbor_reg_weight'] * lg_neighbor_reg_loss
        accuracy = get_seq_edit_accuracy(seq_edit_scores, seq_labels, seq_mask)

        train_loss += total_loss.item()
        train_acc += accuracy

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args['max_clip'])
        optimizer.step()

        if (batch_id + 1) % args['print_every'] == 0:
            print('\repoch %d/%d, batch %d/%d, loss: %.4f, accuracy: %.4f' % (epoch + 1, args['epochs'], batch_id + 1, len(
                train_data), train_loss/(batch_id + 1), train_acc/(batch_id + 1)), end='', flush=True)

    train_loss = float('%.4f' % (train_loss/len(train_data)))
    train_acc = float('%.4f' % (train_acc/len(train_data)))
    print('\nepoch %d/%d, train loss: %.4f, train accuracy: %.4f' %
          (epoch + 1, args['epochs'], train_loss, train_acc))

    return train_loss, train_acc


def test(model, valid_data):
    model.eval()
    total_accuracy = 0.0
    first_step_accuracy = 0.0
    with torch.no_grad():
        for batch_id, batch_data in enumerate(valid_data):
            prod_smi_batch, edits_batch, edits_atom_batch, rxn_classes = batch_data
            for idx, prod_smi in enumerate(prod_smi_batch):
                if rxn_classes is None:
                    edits, edits_atom = model.predict(prod_smi)
                else:
                    edits, edits_atom = model.predict(
                        prod_smi, rxn_class=rxn_classes[idx])
                if edits == edits_batch[idx] and edits_atom == edits_atom_batch[idx]:
                    total_accuracy += 1.0
                if edits[0] == edits_batch[idx][0] and edits_atom[0] == edits_atom_batch[idx][0]:
                    first_step_accuracy += 1.0
    valid_acc = float('%.4f' % (total_accuracy/len(valid_data)))
    valid_first_step_acc = float(
        '%.4f' % (first_step_accuracy/len(valid_data)))

    return valid_acc, valid_first_step_acc


def main(args):
    if args.get('use_rxn_class', False):
        out_dir = os.path.join(ROOT_DIR, 'experiments',
                               args['dataset'], 'with_rxn_class', DATE_TIME)
    else:
        out_dir = os.path.join(ROOT_DIR, 'experiments',
                               args['dataset'], 'without_rxn_class', DATE_TIME)
    os.makedirs(out_dir, exist_ok=True)

    logs_filename = os.path.join(out_dir, 'logs.csv')
    csv_logger = CSVLogger(
        args=args,
        fieldnames=['epoch', 'train_acc', 'valid_acc',
                    'valid_first_step_acc', 'train_loss'],
        filename=logs_filename,
    )

    data_dir = os.path.join(ROOT_DIR, 'data', args['dataset'])
    # load bond, atom and lg vocab
    bond_vocab_file = os.path.join(data_dir, 'train', 'bond_vocab.txt')
    atom_vocab_file = os.path.join(data_dir, 'train', 'atom_lg_vocab.txt')
    bond_vocab = Vocab(joblib.load(bond_vocab_file))
    atom_vocab = Vocab(joblib.load(atom_vocab_file))
    args['delete_bond_indices'] = get_delete_bond_indices(bond_vocab)

    if args.get('use_rxn_class', False):
        train_dir = os.path.join(data_dir, 'train', 'with_rxn_class')
    else:
        train_dir = os.path.join(data_dir, 'train', 'without_rxn_class')
    eval_dir = os.path.join(data_dir, 'valid')

    train_dataset = RetroEditDataset(data_dir=train_dir)
    train_data = train_dataset.loader(
        batch_size=1, num_workers=args['num_workers'], shuffle=True)

    valid_dataset = RetroEvalDataset(
        data_dir=eval_dir, data_file='valid.file.kekulized', use_rxn_class=args['use_rxn_class'])
    valid_data = valid_dataset.loader(
        batch_size=1, num_workers=args['num_workers'])

    model_config = build_model_config(args)

    model = Graph2Edits(config=model_config, atom_vocab=atom_vocab,
                        bond_vocab=bond_vocab, device=DEVICE)
    print(f'Converting model to device: {DEVICE}')
    sys.stdout.flush()
    model.to(DEVICE)
    lg_adj_file = os.path.join(data_dir, 'train', 'lg_cooccurrence_adj.pt')
    model.set_lg_cooccurrence_adj(load_lg_cooccurrence_adj(lg_adj_file, len(atom_vocab)))
    edit_class_stats = load_or_build_edit_class_stats(data_dir, bond_vocab, atom_vocab)
    model.set_edit_class_statistics(edit_class_stats, initialize_lg_tail=True)
    print("Param Count: ", sum([x.nelement()
          for x in model.parameters()]) / 10**6, "M")
    print()

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    mse_loss = nn.MSELoss(reduction='mean')
    optimizer = Adam(model.parameters(), lr=args['lr'])
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=args['patience'], factor=args['factor'], threshold=args['thresh'], threshold_mode='abs')

    best_acc = 0
    for epoch in range(args['epochs']):
        train_loss, train_acc = train_epoch(
            args, epoch, model, train_data, loss_fn, bce_loss, mse_loss, optimizer)
        valid_acc, valid_first_step_acc = test(model, valid_data)
        scheduler.step(valid_acc)
        print('epoch %d/%d, validation accuracy: %.4f, validation_first_acc: %.4f' %
              (epoch + 1, args['epochs'], valid_acc, valid_first_step_acc))
        print('---------------------------------------------------------')
        print()

        row = {
            'epoch': str(epoch + 1),
            'train_acc': str(train_acc),
            'valid_acc': str(valid_acc),
            'valid_first_step_acc': str(valid_first_step_acc),
            'train_loss': str(train_loss),
        }
        csv_logger.writerow(row)

        # update the best accuracy for saving checkpoints
        if valid_acc >= best_acc:
            print(
                f'Best eval accuracy so far. Saving best model from epoch {epoch + 1} (acc={valid_acc})')
            print('---------------------------------------------------------')
            print()
            save_checkpoint(model, out_dir, epoch)
            best_acc = valid_acc

    csv_logger.close()
    print('Experiment finished!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='uspto_50k',
                        help='dataset: uspto_50k or uspto_full')
    parser.add_argument('--use_rxn_class', default=False,
                        action='store_true', help='Whether to use rxn_class')
    parser.add_argument('--atom_message', default=False, action='store_true',
                        help='Node-level or Bond-level message passing')
    parser.add_argument('--use_attn', default=False,
                        action='store_true', help='Whether to use global attention')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of heads in Multihead attention')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maximum number of epochs for training')
    parser.add_argument('--mpn_size', type=int,
                        default=256, help='MPN hidden_dim')
    parser.add_argument('--depth', type=int, default=10,
                        help='Number of iterations')
    parser.add_argument('--dropout_mpn', type=float,
                        default=0.15, help='MPN dropout rate')
    parser.add_argument('--mlp_size', type=int,
                        default=512, help='MLP hidden_dim')
    parser.add_argument('--dropout_mlp', type=float,
                        default=0.2, help='MLP dropout rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

    parser.add_argument('--patience', type=int, default=5,
                        help='Number of epochs with no improvement after which lr will be reduced')
    parser.add_argument('--factor', type=float, default=0.8,
                        help='Factor by which the lr will be reduced')
    parser.add_argument('--thresh', type=float, default=0.01,
                        help='Threshold for measuring the new optimum')
    parser.add_argument('--max_clip', type=int, default=10,
                        help='Maximum number of gradient clip')
    parser.add_argument('--print_every', type=int,
                        default=200, help='Print during train process')
    parser.add_argument('--num_workers', type=int, default=5,
                        help='Number of processes for data loading')
    parser.add_argument('--bond_break_loss_weight', type=float, default=0.5,
                        help='Weight for the auxiliary binary bond-breaking BCE loss')
    parser.add_argument('--bond_energy_loss_alpha', type=float, default=1.0,
                        help='Extra cost-sensitive weight for low-energy bond errors')
    parser.add_argument('--graph_alignment_loss_weight', type=float, default=0.5,
                        help='Weight for product-reactant graph representation MSE loss')
    parser.add_argument('--disable_logit_adjustment', default=False, action='store_true',
                        help='Disable posterior logit adjustment for edit and LG heads')
    parser.add_argument('--logit_adjust_train', default=True, action='store_true',
                        help='Also apply logit adjustment during training CE; default keeps it post-hoc for eval/predict only')
    parser.add_argument('--logit_adjust_tau', type=float, default=0.1,
                        help='Posterior logit-adjustment strength for the flattened edit head')
    parser.add_argument('--lg_logit_adjust_tau', type=float, default=0.1,
                        help='Posterior logit-adjustment strength for LG labels inside the atom/LG head')
    parser.add_argument('--class_prior_smoothing', type=float, default=1.0,
                        help='Additive smoothing for class priors used by logit adjustment')
    parser.add_argument('--disable_class_balanced_loss', default=False, action='store_true',
                        help='Disable class-balanced effective-number weighting for edit loss')
    parser.add_argument('--class_balanced_beta', type=float, default=0.999,
                        help='Beta for class-balanced effective-number edit weights')
    parser.add_argument('--lg_class_balanced_beta', type=float, default=0.999,
                        help='Beta for class-balanced effective-number LG weights')
    parser.add_argument('--class_balanced_warmup_epochs', type=int, default=5,
                        help='Linearly warm up class-balanced edit weights from 1.0 during early epochs')
    parser.add_argument('--class_weight_min', type=float, default=0.5,
                        help='Lower clamp for scheduled class-balanced edit weights')
    parser.add_argument('--class_weight_max', type=float, default=6.0,
                        help='Upper clamp for scheduled class-balanced edit weights')
    parser.add_argument('--lg_gcn_dim', type=int, default=0,
                            help='LG label embedding dimension; 0 picks a model-size aware default')
    parser.add_argument('--lg_gcn_dropout', type=float, default=0.1,
                        help='Dropout used in the LG co-occurrence label GCN')
    parser.add_argument('--lg_head_threshold', type=float, default=50.0,
                        help='Minimum training count for high-frequency LG anchors')
    parser.add_argument('--lg_tail_threshold', type=float, default=10.0,
                        help='Maximum training count for tail LG labels regularized by anchors')
    parser.add_argument('--lg_neighbor_init_alpha', type=float, default=0.2,
                        help='Interpolation factor for initializing tail LG embeddings from high-frequency neighbors')
    parser.add_argument('--lg_neighbor_reg_weight', type=float, default=0.003,
                        help='Weight for LG tail-to-head co-occurrence neighbor embedding regularization')
    args = parser.parse_args().__dict__
    main(args)
