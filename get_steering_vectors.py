import numpy as np
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

def mean_difference(num_layers, wise_activations, labels):
    wise_activations = wise_activations.astype(np.float64)
    directions = []

    for layer in range(num_layers):

        true_mass_mean = np.mean(wise_activations[:, layer, :][labels == 1], axis=0)
        false_mass_mean = np.mean(wise_activations[:, layer, :][labels == 0], axis=0)
        direction = true_mass_mean - false_mass_mean
        # direction = direction.astype(np.float64)
        # direction = direction / np.linalg.norm(direction)
        # direction = direction.squeeze()
        directions.append(direction)
    directions = np.array(directions)

    return directions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='llama2_chat_13B', choices=HF_NAMES.keys(), help='model name')
    parser.add_argument('--dataset_name', type=str, default='bbqa', help='dataset to debias')

    args = parser.parse_args()

    num_layers = NUM_LAYERS[args.model_name]

    layer_wise_activations = np.load(
        f"activations/{args.dataset_name}/{args.model_name}/layer_wise.npy")[:, 1:, :]
    mlp_wise_activations = np.load(f"activations/{args.dataset_name}/{args.model_name}/mlp_wise.npy")
    labels = np.load(f"activations/{args.dataset_name}/{args.model_name}/labels.npy")
    print(len(labels))
    layer_directions = mean_difference(num_layers, layer_wise_activations, labels)
    mlp_directions = mean_difference(num_layers, mlp_wise_activations, labels)

    os.makedirs(f'vectors', exist_ok=True)
    print("Saving layer wise directions")
    np.save(f'vectors/{args.model_name}_layer_wise.npy',
            layer_directions)

    print("Saving mlp wise directions")
    np.save(f'vectors/{args.model_name}_mlp_wise.npy', mlp_directions)


if __name__ == "__main__":
    main()
