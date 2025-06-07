import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
import numpy as np
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import joblib
import matplotlib.pyplot as plt
from nethook import Trace, TraceDict
from functools import partial
from datasets import load_dataset

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

def format_bbq(dataset, idx):
    choices = ['A', 'B', 'C']
    context = dataset[idx]['context']
    question = dataset[idx]['question']
    answers = [dataset[idx]['ans0'], dataset[idx]['ans1'], dataset[idx]['ans2']]
    prompt = f'{context} {question}'
    for i in range(len(answers)):
        prompt += f'\n({choices[i]}) {answers[i]}'
    prompt += '\nAnswer:'

    return prompt

def bbq_evaluate(model, tokenizer, baseline=True, verbose=False, interventions=None, intervention_fn=None):
    print(f"----------Evaluate BBQ-------------")

    frame = load_dataset('json', data_files=f'generate_data/BBQ_test.json')['train']

    def id(head_output, layer_name):
        return head_output

    cors = []

    for i in range(len(frame)):
        label = frame[i]['label']
        prompt_end = format_bbq(frame, i)
        prompt = prompt_end

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs.input_ids
        # attention_mask = inputs.attention_mask
        max_len = input_ids.shape[-1] + 100

        if interventions == None:
            intervene = id
            layers_to_intervene = []
        else:
            intervene = partial(intervention_fn, start_edit_location='lt')
            layers_to_intervene = list(interventions.keys())

        with torch.no_grad():
            if baseline:
                # model_gen_tokens = model.generate(input_ids=input_ids, top_k=1, max_length=max_len, num_return_sequences=1,)[:, input_ids.shape[-1]:]
                logits = model(input_ids=input_ids).logits[0, -1]
            else:
                with TraceDict(model, layers_to_intervene, edit_output=intervene) as ret:
                    # model_gen_tokens = model.generate(input_ids=input_ids, top_k=1, max_length=max_len, num_return_sequences=1,)[:, input_ids.shape[-1]:]
                    logits = model(input_ids=input_ids).logits[0, -1]
        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("Answer: A").input_ids[-1]],
                        logits[tokenizer("Answer: B").input_ids[-1]],
                        logits[tokenizer("Answer: C").input_ids[-1]],
                    ]
                ).float(),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        pred = np.argmax(probs)
        if verbose:
            print("MODEL_OUTPUT: ", {0: "A", 1: "B", 2: "C"}[pred])
        cor = pred == label
        cors.append(cor)

    accuracy = np.mean(cors)
    print(accuracy)
    torch.cuda.empty_cache()

    return accuracy

def plot_acc():
    model_name = 'llama2_chat_13B'
    alpha = 1
    with open(f"bias_bench/results/ablation_layer/{model_name}_alpha_{alpha}.json", 'r') as f:
        result_dict = json.load(f)
    plt.figure(figsize=(12, 8))
    accuracy = result_dict['accs']
    acc_base = result_dict['acc base']
    plt.plot(range(0, len(accuracy)), accuracy, color='red', linewidth=2)
    plt.axhline(y=acc_base, color='red', linestyle='--')

    # 设置标题和标签
    plt.title(f'Accuracy for Different Layer - Model: {model_name}', fontsize=18)
    plt.xlabel('Layer', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(np.arange(0, len(accuracy), 1))
    plt.xlim(0, len(accuracy) - 1)

    # 展示图像
    plt.tight_layout()
    plt.show()


def get_interventions_dict(component, layers_to_intervention, vectors, probes):

    interventions = {}
    for layer in layers_to_intervention:
        direction = vectors[layer, :]
        # direction = direction / np.linalg.norm(direction)
        probe = probes[layer]
        if component == 'layer':
            interventions[f"model.layers.{layer}"] = {}
            interventions[f"model.layers.{layer}"]['direction'] = direction.squeeze()
            interventions[f"model.layers.{layer}"]['probe'] = probe
        elif component == 'mlp':
            interventions[f"model.layers.{layer}.mlp"] = {}
            interventions[f"model.layers.{layer}.mlp"]['direction'] = direction.squeeze()
            interventions[f"model.layers.{layer}.mlp"]['probe'] = probe
    return interventions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='llama2_chat_13B', help='model name')
    parser.add_argument("--model_dir", type=str, default="llama2_chat_13B", help='local directory with model data')
    parser.add_argument('--alpha', type=float, default=1, help='alpha, intervention strength')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--component', type=str, default='layer')


    args = parser.parse_args()

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # create model
    MODEL = HF_NAMES[args.model_name] if not args.model_dir else args.model_dir

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                 device_map='auto')

    tokenizer.pad_token = tokenizer.eos_token
    
    num_layers = NUM_LAYERS[args.model_name]

    acc_base = bbq_evaluate(
        model=model,
        tokenizer=tokenizer,
        verbose=False,
        baseline=True
    )

    vectors = np.load(f"vectors/{args.model_name}_{args.component}_wise.npy")

    probes = joblib.load(f'probes/{args.model_name}_{args.component}_wise.pkl')

    if args.component == 'layer':
        def lt_modulated_vector_add(layer_output, layer_name, start_edit_location='lt'):
            direction = interventions[layer_name]['direction']
            direction_to_add = torch.tensor(direction).to(layer_output[0].device.index)
            probe = interventions[layer_name]['probe']
            # print(f"BEFORE first_edit_done: {state.first_edit_done}, apply_edit: {state.apply_edit}")
            if start_edit_location == 'lt':
                layer_output_np = layer_output[0][:, -1, :].cpu().numpy()
                layer_output_np = layer_output_np.reshape(-1, layer_output_np.shape[-1])
                y = probe.predict(layer_output_np)
                # print(y)
                if y[0] == 0:
                    layer_output[0][:, -1, :] += args.alpha * direction_to_add
            else:
                layer_output_np = layer_output[0][:, start_edit_location:, :].cpu().numpy()
                layer_output_np = layer_output_np.reshape(-1, layer_output_np.shape[-1])
                y = probe.predict(layer_output_np)
                # print(y)
                mask = torch.tensor(y == 0).to(layer_output[0].device.index)
                layer_output[0][:, start_edit_location:, :] += mask[:, None] * args.alpha * direction_to_add
                # layer_output[0][:, start_edit_location:, :] += args.alpha * direction_to_add
            return layer_output
    else:
        def lt_modulated_vector_add(layer_output, layer_name):
            return layer_output
    accs = []
    for layer in tqdm(range(0, num_layers)):
        vector = vectors[layer, :]
        print(f'Layer{layer}: {np.linalg.norm(vector)}')
        interventions = get_interventions_dict(args.component, [layer], vectors, probes)
        acc = bbq_evaluate(
            model=model,
            tokenizer=tokenizer,
            verbose=False,
            baseline=False,
            interventions=interventions,
            intervention_fn=lt_modulated_vector_add
        )
        accs.append(acc)
        
    result_dict = {'acc base': acc_base, 'accs': accs}

    os.makedirs(f"bias_bench/results/ablation_layer", exist_ok=True)
    with open(f"bias_bench/results/ablation_layer/{args.model_name}_alpha_{args.alpha}.json", "w") as f:
        json.dump(result_dict, f)

    plt.figure(figsize=(12, 8))
    accuracy = accs
    plt.plot(range(0, len(accuracy)), accuracy, color='red', linewidth=2)
    plt.axhline(y=acc_base, color='red', linestyle='--')

    # 设置标题和标签
    plt.title(f'Accuracy for Different Layer - Model: {args.model_name}', fontsize=18)
    plt.xlabel('Layer', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(np.arange(0, len(accuracy), 1))
    plt.xlim(0, len(accuracy) - 1)

    # 展示图像
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    # plot_acc()
