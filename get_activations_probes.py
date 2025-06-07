import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from utils import get_llama_activations_bau, tokenized_bbq, tokenized_bbq_fs, tokenized_mmlu
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


HF_NAMES = {
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b',
    'vicuna_13B': 'lmsys/vicuna-13b-v1.5',
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf',
    'llama3_8B_instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'mistral_7B_instruct': 'mistralai/Mistral-7B-Instruct-v0.3',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2_chat_13B')
    parser.add_argument("--model_dir", type=str, default="llama2_chat_13B", help='local directory with model data')
    parser.add_argument('--dataset', type=str, default='bbq_fs')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--seed', type=int, default=42, help='seed')

    args = parser.parse_args()
    MODEL = HF_NAMES[args.model_name] if not args.model_dir else args.model_dir

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16,  device_map='auto')

    print("Tokenizing prompts")

    if args.dataset == 'bbq_zs':
        dataset = load_dataset("oskarvanderwal/bbq", 'All')['test']
        prompts, labels, _, positions, _, _ = tokenized_bbq(dataset, tokenizer, model)
    elif args.dataset == 'bbq_fs':
        dataset = load_dataset("oskarvanderwal/bbq", 'All')['test']
        prompts, labels, _, positions, _, _ = tokenized_bbq_fs(dataset, tokenizer, model)
    elif args.dataset == 'mmlu':
        dataset = load_dataset('json', data_files='generate_data/mmlu.json')['train']
        prompts, labels, positions = tokenized_mmlu(dataset, tokenizer)

    all_layer_wise_activations = []
    all_mlp_wise_activations = []

    print("Getting activations")
    for i, prompt in enumerate(tqdm(prompts)):
        layer_wise_activations, mlp_wise_activations = get_llama_activations_bau(model, prompt)
        all_layer_wise_activations.append(layer_wise_activations[:, positions[i], :].copy())
        all_mlp_wise_activations.append(mlp_wise_activations[:, positions[i], :].copy())

    os.makedirs(f'activations/{args.dataset}/{args.model_name}', exist_ok=True)

    print("Saving labels")
    np.save(f'activations/{args.dataset}/{args.model_name}/labels.npy', labels)

    print("Saving layer wise activations")
    np.save(f'activations/{args.dataset}/{args.model_name}/layer_wise.npy', all_layer_wise_activations)

    print("Saving mlp wise activations")
    np.save(f'activations/{args.dataset}/{args.model_name}/mlp_wise.npy', all_mlp_wise_activations)


if __name__ == '__main__':
    main()