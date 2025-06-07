import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import torch
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import bbq_evaluate, unqover_evaluate, crows_evaluate, mmlu_evaluate, arc_evaluate, obqa_evaluate
import joblib
from perplexity import compute_perplexity

HF_NAMES = {
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b',
    'vicuna_13B': 'lmsys/vicuna-13b-v1.5',
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf',
    'llama3_8B_instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'mistral_7B_instruct': 'mistralai/Mistral-7B-Instruct-v0.3',
}

CROWS_CATEGORY = ['age', 'disability', 'gender', 'nationality', 'physical-appearance', 'race', 'religion', 'socioeconomic', 'sexual-orientation']


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
    parser.add_argument("--model_name", type=str, default='llama2_chat_13B', choices=HF_NAMES.keys(), help='model name')
    parser.add_argument("--model_dir", type=str, default="llama2_chat_13B", help='local directory with model data')
    parser.add_argument('--alpha', type=float, default=1, help='alpha, intervention strength')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--baseline_mode', type=bool, default=False)
    parser.add_argument('--layers_to_intervention', type=list, default=[15])
    parser.add_argument('--component', type=str, default='layer')

    args = parser.parse_args()

    # set seeds
    np.random.seed(args.seed)

    # create model
    MODEL = HF_NAMES[args.model_name] if not args.model_dir else args.model_dir

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16,  device_map='auto')

    tokenizer.pad_token = tokenizer.eos_token

    if args.baseline_mode:
        tag = f'{args.model_name}_baseline'
        crows_evaluate(tag=tag,
                       component=args.component,
                       model=model,
                       tokenizer=tokenizer,
                       baseline_mode=True,
                       persistent_dir="bias_bench",
                       is_generative=True,
                       bias_type=None
                       )
        # for category in CROWS_CATEGORY:
        #     crows_evaluate(tag=tag,
        #                    component=args.component,
        #                    model=model,
        #                    tokenizer=tokenizer,
        #                    baseline_mode=True,
        #                    persistent_dir="bias_bench",
        #                    is_generative=True,
        #                    bias_type=category
        #                    )
        bbq_evaluate(
            tag=tag,
            component=args.component,
            model=model,
            tokenizer=tokenizer,
            bias_type='all',
            persistent_dir='bias_bench',
            verbose=False,
            baseline=True,
            few_shot=False
        )
        unqover_evaluate(
            tag=tag,
            component=args.component,
            model=model,
            tokenizer=tokenizer,
            few_shot=False,
            persistent_dir='bias_bench',
            device="cuda",
            verbose=False,
            baseline=True
        )
        bbq_evaluate(
            tag=tag,
            component=args.component,
            model=model,
            tokenizer=tokenizer,
            bias_type='all',
            persistent_dir='bias_bench',
            verbose=False,
            baseline=True,
            few_shot=True
        )
        unqover_evaluate(
            tag=tag,
            component=args.component,
            model=model,
            tokenizer=tokenizer,
            few_shot=True,
            persistent_dir='bias_bench',
            device="cuda",
            verbose=False,
            baseline=True
        )

        compute_perplexity(tag, model, tokenizer, baseline_mode=True)
        mmlu_evaluate(
            tag=tag,
            component=args.component,
            model=model,
            tokenizer=tokenizer,
            persistent_dir="bias_bench",
            category='all',
            baseline=True,
        )
        for category in ['easy', 'challenge']:
            arc_evaluate(
                tag=tag,
                model=model,
                tokenizer=tokenizer,
                persistent_dir="bias_bench",
                category=category,
                baseline=True,
            )
        obqa_evaluate(
            tag=tag,
            model=model,
            tokenizer=tokenizer,
            persistent_dir="bias_bench",
            baseline=True,
        )

    else:

        tag = f'{args.model_name}_itb_alpha_{args.alpha}'

        vectors = np.load(f"vectors/{args.model_name}_{args.component}_wise.npy")

        probes = joblib.load(f'probes/{args.model_name}_{args.component}_wise.pkl')

        print("Layers intervened: ", args.layers_to_intervention)
        for layer in args.layers_to_intervention:
            vector = vectors[layer, :]
            print(f'Layer{layer}: {np.linalg.norm(vector)}')
            # tag += f"_{layer}"

        interventions = get_interventions_dict(args.component, args.layers_to_intervention, vectors, probes)
        def lt_modulated_vector_add(layer_output, layer_name, start_edit_location='lt'):
            direction = interventions[layer_name]['direction']
            direction_to_add = torch.tensor(direction).to(layer_output[0].device.index)
            probe = interventions[layer_name]['probe']
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

        crows_evaluate(tag=tag,
                       component=args.component,
                       model=model,
                       tokenizer=tokenizer,
                       baseline_mode=False,
                       persistent_dir="bias_bench",
                       is_generative=True,
                       bias_type=None,
                       interventions=interventions,
                       intervention_fn=lt_modulated_vector_add,
                       )
        # for category in CROWS_CATEGORY:
        #     crows_evaluate(tag=tag,
        #                    component=args.component,
        #                    model=model,
        #                    tokenizer=tokenizer,
        #                    baseline_mode=False,
        #                    persistent_dir="bias_bench",
        #                    is_generative=True,
        #                    bias_type=category,
        #                    interventions=interventions,
        #                    intervention_fn=lt_modulated_vector_add,
        #                    )
        bbq_evaluate(
            tag=tag,
            component=args.component,
            model=model,
            tokenizer=tokenizer,
            bias_type='all',
            persistent_dir='bias_bench',
            verbose=False,
            baseline=False,
            few_shot=False,
            interventions=interventions,
            intervention_fn=lt_modulated_vector_add
        )
        unqover_evaluate(
            tag=tag,
            component=args.component,
            model=model,
            tokenizer=tokenizer,
            few_shot=False,
            persistent_dir='bias_bench',
            device="cuda",
            verbose=False,
            baseline=False,
            interventions=interventions,
            intervention_fn=lt_modulated_vector_add,
        )
        bbq_evaluate(
            tag=tag,
            component=args.component,
            model=model,
            tokenizer=tokenizer,
            bias_type='all',
            persistent_dir='bias_bench',
            verbose=False,
            baseline=False,
            few_shot=True,
            interventions=interventions,
            intervention_fn=lt_modulated_vector_add
        )
        unqover_evaluate(
            tag=tag,
            component=args.component,
            model=model,
            tokenizer=tokenizer,
            few_shot=True,
            persistent_dir='bias_bench',
            device="cuda",
            verbose=False,
            baseline=False,
            interventions=interventions,
            intervention_fn=lt_modulated_vector_add,
        )

        compute_perplexity(tag, model, tokenizer, baseline_mode=False, interventions=interventions,
                           intervention_fn=lt_modulated_vector_add)

        mmlu_evaluate(
            tag=tag,
            component=args.component,
            model=model,
            tokenizer=tokenizer,
            category='all',
            baseline=False,
            persistent_dir="bias_bench",
            interventions=interventions,
            intervention_fn=lt_modulated_vector_add,
        )
        for category in ['easy', 'challenge']:
            arc_evaluate(
                tag=tag,
                model=model,
                tokenizer=tokenizer,
                persistent_dir="bias_bench",
                category=category,
                baseline=False,
                interventions=interventions,
                intervention_fn=lt_modulated_vector_add,
            )
        obqa_evaluate(
            tag=tag,
            model=model,
            tokenizer=tokenizer,
            persistent_dir="bias_bench",
            baseline=False,
            interventions=interventions,
            intervention_fn=lt_modulated_vector_add,
        )



if __name__ == "__main__":
    main()
