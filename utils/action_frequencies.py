import os
import sys
import joblib
from collections import Counter

# 将项目根目录加入 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def build_action_freq_map(dataset='uspto_50k'):
    train_file = f'data/{dataset}/train/train.file.kekulized'
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")

    rxns_data = joblib.load(train_file)
    action_counter = Counter()

    for rxn in rxns_data:
        for edit in rxn.edits:
            if edit != 'Terminate':
                action_counter[edit] += 1

    save_path = f'data/{dataset}/train/action_freq_map.pkl'
    joblib.dump(action_counter, save_path)
    print(f"Saved action frequency map with {len(action_counter)} unique actions to {save_path}")

    freqs = list(action_counter.values())
    if freqs:
        print(f"Min freq: {min(freqs)}, Max freq: {max(freqs)}, Median: {sorted(freqs)[len(freqs)//2]}")
    return action_counter

if __name__ == '__main__':
    build_action_freq_map(dataset='uspto_50k')
