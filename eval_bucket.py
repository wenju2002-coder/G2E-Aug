import numpy as np
import os
import argparse
import csv
import joblib
from tqdm import tqdm
from collections import defaultdict, Counter
import torch
from rdkit import Chem, RDLogger

from models import Graph2Edits, BeamSearch
from utils.class_balance import load_lg_cooccurrence_adj, load_or_build_edit_class_stats
from utils.action_frequencies import build_action_freq_map  # 确保训练时已生成

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
    except:
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
    return Chem.MolToSmiles(p_mol)

# ---------- 分桶相关 ----------
def get_bucket(edits, freq_map):
    """
    edits: list of edit tuples, e.g. [('Change Atom', (1,0)), ('Terminate',)]
    freq_map: Counter mapping edit -> frequency in training
    Returns: 'Many-shot', 'Medium-shot', or 'Few-shot'
    """
    min_freq = float('inf')
    for edit in edits:
        if edit == 'Terminate':
            continue
        f = freq_map.get(edit, 0)  # 未见过的编辑视为 0
        if f < min_freq:
            min_freq = f
    # 如果 min_freq 仍为 inf（只有 Terminate），视为 Many-shot 吧，但实际不会发生
    if min_freq > 100:
        return 'Many-shot'
    elif min_freq >= 20:
        return 'Medium-shot'
    else:
        return 'Few-shot'

# ---------- Round-trip 辅助（从 eval-1.py 移植）----------
def smi_tokenizer(smi):
    import re
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

def round_trip_translate(pred_text, opt, translator):
    """Run round-trip translation given predicted SMILES file."""
    all_scores, all_predictions = translator.translate(
        src_path=pred_text,
        tgt_path=opt.tgt,
        src_dir=opt.src_dir,
        batch_size=opt.batch_size,
        attn_debug=opt.attn_debug
    )
    return all_predictions

# ---------- 主评估函数 ----------
def get_available_report_path(exp_dir, epoch_name):
    epoch_tag = os.path.splitext(os.path.basename(epoch_name))[0]
    report_path = os.path.join(exp_dir, f'bucket_eval_{epoch_tag}.md')
    if not os.path.exists(report_path):
        return report_path

    file_idx = 1
    while True:
        candidate = os.path.join(exp_dir, f'bucket_eval_{epoch_tag}_{file_idx}.md')
        if not os.path.exists(candidate):
            return candidate
        file_idx += 1


def get_available_csv_path(exp_dir, epoch_name):
    epoch_tag = os.path.splitext(os.path.basename(epoch_name))[0]
    csv_path = os.path.join(exp_dir, f'bucket_eval_{epoch_tag}.csv')
    if not os.path.exists(csv_path):
        return csv_path

    file_idx = 1
    while True:
        candidate = os.path.join(exp_dir, f'bucket_eval_{epoch_tag}_{file_idx}.csv')
        if not os.path.exists(candidate):
            return candidate
        file_idx += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='uspto_50k',
                        help='dataset: uspto_50k or uspto_full')
    parser.add_argument("--use_rxn_class", default=False,
                        action='store_true', help='Whether to use rxn_class')
    parser.add_argument('--experiments', type=str, default="16-07-2026--01-17-31",
                        help='Name of edits prediction experiment (subfolder under with_rxn_class/without_rxn_class)')
    parser.add_argument('--epoch', type=str, default='epoch_86.pt',
                        help='Checkpoint file name, e.g. epoch_105.pt')
    parser.add_argument('--report_file', type=str, default=None,
                        help='Optional report filename saved under the experiment directory')
    parser.add_argument('--csv_file', type=str, default=None,
                        help='Optional CSV filename saved under the experiment directory')
    parser.add_argument('--beam_size', type=int, default=10,
                        help='Beam search width')
    parser.add_argument('--max_steps', type=int, default=9,
                        help='maximum number of edit steps')
    parser.add_argument('--round_trip', action='store_true',
                        help='Enable round-trip evaluation (requires separate translator)')
    parser.add_argument('--rt_model', type=str, default=None,
                        help='Path to round-trip translation model (optional)')
    args = parser.parse_args()
    args.dataset = args.dataset.lower()

    # 1. 加载动作频率映射
    freq_map_path = f'data/{args.dataset}/train/action_freq_map.pkl'
    if not os.path.exists(freq_map_path):
        print("Frequency map not found. Building now...")
        from utils.action_frequencies import build_action_freq_map
        freq_map = build_action_freq_map(dataset=args.dataset)
    else:
        freq_map = joblib.load(freq_map_path)
    print(f"Loaded frequency map with {len(freq_map)} actions.")

    # 2. 加载测试数据
    data_dir = os.path.join(ROOT_DIR, 'data', args.dataset, 'test')
    test_file = os.path.join(data_dir, 'test.file.kekulized')
    test_data = joblib.load(test_file)

    # 3. 确定实验目录
    if args.use_rxn_class:
        exp_dir = os.path.join(ROOT_DIR, 'experiments', args.dataset, 'with_rxn_class', args.experiments)
    else:
        exp_dir = os.path.join(ROOT_DIR, 'experiments', args.dataset, 'without_rxn_class', args.experiments)

    # 4. 加载模型
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

    # 5. 初始化 beam search
    beam_model = BeamSearch(model=model, step_beam_size=10,
                            beam_size=args.beam_size, use_rxn_class=args.use_rxn_class)

    # 6. 初始化统计字典: stats[(rxn_class, bucket)] = {'total':0, 'top1':0, ..., 'rt':0}
    stats = defaultdict(lambda: {'total': 0, 'top1': 0, 'top3': 0, 'top5': 0, 'top10': 0, 'rt': 0})
    # 另外分别统计各桶的总样本数（用于整体汇总）
    bucket_total = defaultdict(int)

    # 如果启用 round-trip，需准备 translator（这里仅为示例，实际需构建）
    translator = None
    if args.round_trip:
        # 假设你有独立的翻译模型（如 OpenNMT），此处简化
        from onmt.translate.translator import build_translator
        from onmt.utils.logging import init_logger
        import onmt.opts
        # 设置解析参数，这里仅示意，你需要根据实际配置调整
        rt_parse = argparse.ArgumentParser()
        onmt.opts.add_md_help_argument(rt_parse)
        onmt.opts.translate_opts(rt_parse)
        rt_opt = rt_parse.parse_args(['-model', args.rt_model, '-gpu', '0'])
        logger = init_logger(rt_opt.log_file)
        translator = build_translator(rt_opt, report_score=True)

    # 7. 循环每个测试样本
    p_bar = tqdm(enumerate(test_data), total=len(test_data))
    for idx, rxn_data in p_bar:
        rxn_smi = rxn_data.rxn_smi
        rxn_class = rxn_data.rxn_class  # 可能为 None
        true_edits = rxn_data.edits
        bucket = get_bucket(true_edits, freq_map)
        bucket_total[bucket] += 1

        # 解析反应物和产物
        r, p = rxn_smi.split('>>')
        r_mol = Chem.MolFromSmiles(r)
        [a.ClearProp('molAtomMapNumber') for a in r_mol.GetAtoms()]
        r_mol = Chem.MolFromSmiles(Chem.MolToSmiles(r_mol))
        r_smi = Chem.MolToSmiles(r_mol, isomericSmiles=True)
        r_set = set(r_smi.split('.'))

        # 运行 beam search
        with torch.no_grad():
            top_k_results = beam_model.run_search(
                prod_smi=p, max_steps=args.max_steps, rxn_class=rxn_class if args.use_rxn_class else None)

        # 检查 Top-1,3,5,10 命中
        beam_matched_idx = None  # 记录命中的最小索引（0-based）
        for beam_idx, path in enumerate(top_k_results):
            pred_smi = path['final_smi']
            if pred_smi == 'final_smi_unmapped':
                continue
            pred_set = set(pred_smi.split('.'))
            if pred_set == r_set:
                beam_matched_idx = beam_idx
                break

        # 更新统计（按 rxn_class 和 bucket）
        if rxn_class is None:
            # 如果没有反应类别，统一归为 class -1
            rxn_class = -1
        key = (rxn_class, bucket)
        stats[key]['total'] += 1
        if beam_matched_idx is not None:
            if beam_matched_idx < 1: stats[key]['top1'] += 1
            if beam_matched_idx < 3: stats[key]['top3'] += 1
            if beam_matched_idx < 5: stats[key]['top5'] += 1
            if beam_matched_idx < 10: stats[key]['top10'] += 1

        # Round-trip 评估（如果启用）
        if args.round_trip and beam_matched_idx is not None:
            # 使用 beam search 中第 0 个预测（或最优预测）做 round-trip
            # 这里简化：我们取第一个预测（top-1）
            best_path = top_k_results[0] if top_k_results else None
            if best_path and best_path['final_smi'] != 'final_smi_unmapped':
                pred_smi = best_path['final_smi']
                # 将预测的 SMILES 保存为临时文件，然后调用 translator
                temp_file = os.path.join(exp_dir, f'rt_temp_{idx}.txt')
                with open(temp_file, 'w') as f:
                    # tokenize
                    tokenized = smi_tokenizer(pred_smi)
                    f.write(tokenized + '\n')
                # 执行翻译（假设 translator 已配置）
                all_predictions = round_trip_translate(temp_file, rt_opt, translator)
                # 检查 round-trip 结果是否与真实反应物集匹配
                rt_success = False
                for preds in all_predictions:
                    for pred in preds:
                        cleaned = ''.join(pred.strip().split(' '))
                        mol = Chem.MolFromSmiles(cleaned)
                        if mol is not None:
                            cleaned_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
                            if set(cleaned_smi.split('.')) == r_set:
                                rt_success = True
                                break
                    if rt_success:
                        break
                if rt_success:
                    stats[key]['rt'] += 1

        # 更新进度描述
        if idx % 10 == 0:
            msg = f"Total: {idx+1}"
            p_bar.set_description(msg)

    # 8. 输出报告（Markdown表格）
    # 先按反应类别排序
    classes = sorted(set(k[0] for k in stats.keys() if k[0] != -1))
    if -1 in set(k[0] for k in stats.keys()):
        classes.append(-1)
    buckets = ['Many-shot', 'Medium-shot', 'Few-shot']

    # 生成整体汇总行
    overall = {b: {'total':0, 'top1':0, 'top3':0, 'top5':0, 'top10':0, 'rt':0} for b in buckets}

    report_lines = [
        f"# Bucket Evaluation: {args.experiments}",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Checkpoint: `{args.epoch}`",
        f"- Beam size: `{args.beam_size}`",
        f"- Max steps: `{args.max_steps}`",
        f"- Unit: `reaction`",
        f"- Test reactions: `{len(test_data)}`",
        f"- Use reaction class: `{args.use_rxn_class}`",
        f"- Round-trip: `{args.round_trip}`",
        "",
        "## Evaluation Report (Bucket by Minimum Edit Frequency)",
        "",
        "| Reaction Class | Bucket | Total Reactions | Top-1 | Top-3 | Top-5 | Top-10 | Round-trip |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    csv_rows = []

    for rxn_cls in classes:
        for bucket in buckets:
            key = (rxn_cls, bucket)
            if key not in stats:
                # 如果没有样本，跳过或打印空行
                continue
            s = stats[key]
            total = s['total']
            if total == 0:
                continue
            top1 = s['top1'] / total * 100
            top3 = s['top3'] / total * 100
            top5 = s['top5'] / total * 100
            top10 = s['top10'] / total * 100
            rt = s['rt'] / total * 100 if args.round_trip else '-'
            cls_name = str(rxn_cls) if rxn_cls != -1 else 'Overall'
            report_lines.append(
                f"| {cls_name} | {bucket} | {total} | {top1:.1f}% | {top3:.1f}% | {top5:.1f}% | {top10:.1f}% | {rt if rt=='-' else f'{rt:.1f}%'} |")
            csv_rows.append({
                'section': 'by_reaction_class',
                'reaction_class': cls_name,
                'bucket': bucket,
                'total_reactions': total,
                'top1': round(top1, 4),
                'top3': round(top3, 4),
                'top5': round(top5, 4),
                'top10': round(top10, 4),
                'round_trip': '' if rt == '-' else round(rt, 4),
            })

            # 累加整体汇总
            overall[bucket]['total'] += total
            overall[bucket]['top1'] += s['top1']
            overall[bucket]['top3'] += s['top3']
            overall[bucket]['top5'] += s['top5']
            overall[bucket]['top10'] += s['top10']
            overall[bucket]['rt'] += s['rt']

    # 打印整体汇总（所有类别合并）
    report_lines.extend([
        "",
        "### Overall Summary (All Reaction Classes)",
        "",
        "| Bucket | Total Reactions | Top-1 | Top-3 | Top-5 | Top-10 | Round-trip |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for bucket in buckets:
        s = overall[bucket]
        total = s['total']
        if total == 0:
            continue
        top1 = s['top1'] / total * 100
        top3 = s['top3'] / total * 100
        top5 = s['top5'] / total * 100
        top10 = s['top10'] / total * 100
        rt = s['rt'] / total * 100 if args.round_trip else '-'
        report_lines.append(
            f"| {bucket} | {total} | {top1:.1f}% | {top3:.1f}% | {top5:.1f}% | {top10:.1f}% | {rt if rt=='-' else f'{rt:.1f}%'} |")
        csv_rows.append({
            'section': 'overall',
            'reaction_class': 'All',
            'bucket': bucket,
            'total_reactions': total,
            'top1': round(top1, 4),
            'top3': round(top3, 4),
            'top5': round(top5, 4),
            'top10': round(top10, 4),
            'round_trip': '' if rt == '-' else round(rt, 4),
        })

    overall_total = sum(overall[bucket]['total'] for bucket in buckets)
    if overall_total != len(test_data):
        print(f'Warning: overall bucket total {overall_total} != test reactions {len(test_data)}. '
              f'This report should count reactions, not edit steps.')

    report_text = '\n'.join(report_lines)
    print('\n' + report_text)

    report_path = os.path.join(exp_dir, args.report_file) if args.report_file else get_available_report_path(exp_dir, args.epoch)
    with open(report_path, 'w', encoding='utf-8') as fp:
        fp.write(report_text + '\n')
    print(f'\nSaved bucket evaluation report to: {report_path}')

    csv_path = os.path.join(exp_dir, args.csv_file) if args.csv_file else get_available_csv_path(exp_dir, args.epoch)
    fieldnames = [
        'section', 'reaction_class', 'bucket', 'total_reactions',
        'top1', 'top3', 'top5', 'top10', 'round_trip'
    ]
    with open(csv_path, 'w', encoding='utf-8', newline='') as fp:
        meta_writer = csv.writer(fp)
        meta_writer.writerow(['experiment', args.experiments])
        meta_writer.writerow(['dataset', args.dataset])
        meta_writer.writerow(['checkpoint', args.epoch])
        meta_writer.writerow(['beam_size', args.beam_size])
        meta_writer.writerow(['max_steps', args.max_steps])
        meta_writer.writerow(['unit', 'reaction'])
        meta_writer.writerow(['num_test_reactions', len(test_data)])
        meta_writer.writerow(['use_rxn_class', args.use_rxn_class])
        meta_writer.writerow(['round_trip_enabled', args.round_trip])
        meta_writer.writerow([])

        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f'Saved bucket evaluation CSV to: {csv_path}')

if __name__ == '__main__':
    main()
