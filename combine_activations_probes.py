import numpy as np
import os

model_name = 'llama2_chat_13B'


all_layer_wise_activations = []
all_mlp_wise_activations = []
all_labels = []

for dataset_name in ['bbq_zs', 'bbq_fs', 'mmlu']:

    layer_wise_activations = np.load(
        f"activations/{dataset_name}/{model_name}/layer_wise.npy")
    mlp_wise_activations = np.load(f"activations/{dataset_name}/{model_name}/mlp_wise.npy")
    labels = np.load(f"activations/{dataset_name}/{model_name}/labels.npy")

    all_layer_wise_activations.extend(layer_wise_activations)
    all_mlp_wise_activations.extend(mlp_wise_activations)
    all_labels.extend(labels)

    print(len(all_layer_wise_activations))
    print(len(all_mlp_wise_activations))
    print(len(all_labels))


dataset_name = 'probes'
os.makedirs(f'activations/{dataset_name}/{model_name}', exist_ok=True)

print("Saving layer wise activations")
np.save(f'activations/{dataset_name}/{model_name}/layer_wise.npy', all_layer_wise_activations)
del all_layer_wise_activations

print("Saving mlp wise activations")
np.save(f'activations/{dataset_name}/{model_name}/mlp_wise.npy', all_mlp_wise_activations)
del all_mlp_wise_activations

print("Saving labels")
np.save(f'activations/{dataset_name}/{model_name}/labels.npy', all_labels)
