import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import numpy as np
import argparse
import joblib
from tqdm import tqdm
from cuml.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

HF_NAMES = {
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b',
    'vicuna_13B': 'lmsys/vicuna-13b-v1.5',
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf',
    'llama3_8B_instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'mistral_7B_instruct': 'mistralai/Mistral-7B-Instruct-v0.3',
}

NUM_LAYERS = {
    'llama2_chat_7B': 32,
    'llama2_chat_13B': 40,
    'llama3_8B_instruct': 32,
    'vicuna_7B': 32,
    'vicuna_13B': 40,
    'mistral_7B_instruct': 32,
}

def train_probes(seed, train_set_idxs, val_set_idxs, activations, labels, num_layers):

    all_accs = []
    probes = []
    all_X_train = activations[train_set_idxs]
    all_X_val = activations[val_set_idxs]
    y_train = labels[train_set_idxs]
    # assert np.sum(y_train == 0) == np.sum(y_train == 1)
    y_val = labels[val_set_idxs]
    # assert np.sum(y_val == 0) == np.sum(y_val == 1)

    for layer in tqdm(range(num_layers), desc="train_probes_with_val"):
        X_train = all_X_train[:,layer,:]
        X_val = all_X_val[:,layer,:]

        # clf = LogisticRegression(random_state=seed, max_iter=10000).fit(X_train, y_train)
        clf = LogisticRegression(max_iter=10000).fit(X_train, y_train)
        y_pred = clf.predict(X_train)
        y_val_pred = clf.predict(X_val)
        all_accs.append(accuracy_score(y_val, y_val_pred))
        probes.append(clf)

    all_accs = np.array(all_accs)

    return probes, all_accs



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='llama2_chat_13B', help='model name')
    parser.add_argument('--dataset_name', type=str, default='probes', help='dataset to debias')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--val_ratio', type=float, default=0.2)

    args = parser.parse_args()

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    num_layers = NUM_LAYERS[args.model_name]


    layer_wise_activations = np.load(
        f"activations/{args.dataset_name}/{args.model_name}/layer_wise.npy")[:, 1:, :]
    # mlp_wise_activations = np.load(f"activations/{args.dataset_name}/{args.model_name}/mlp_wise.npy")
    # head_wise_activations = np.load(f"activations/{args.dataset_name}/{args.model_name}/head_wise.npy")
    labels = np.load(f"activations/{args.dataset_name}/{args.model_name}/labels.npy")
    print(len(labels))

    # head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h=num_heads)

    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    # idx = idx[:size]
    train_set_idxs = idx[:int(len(idx) * (1 - args.val_ratio))]  # 标签 1 的训练集
    val_set_idxs = idx[int(len(idx) * (1 - args.val_ratio)):]

    # np.save(f"activations/{args.dataset_name}/{args.model_name}/train_set_idxs.npy", train_set_idxs)
    # np.save(f"activations/{args.dataset_name}/{args.model_name}/val_set_idxs.npy", val_set_idxs)


    layer_probes, all_layer_accs_np = train_probes(args.seed, train_set_idxs, val_set_idxs, layer_wise_activations, labels, num_layers)

    os.makedirs(f'probes', exist_ok=True)
    print("Saving layer probes")
    joblib.dump(layer_probes, f'probes/{args.model_name}_layer_wise.pkl')

    print("Saving layer wise accuracy")
    np.save(f'probes/{args.model_name}_layer_wise_accuracy.npy',
            all_layer_accs_np)
    print(all_layer_accs_np)


if __name__ == "__main__":
    main()
