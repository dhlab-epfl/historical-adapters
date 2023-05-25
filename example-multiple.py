# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import numpy as np

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA
import pdb

from scripts.prepare_alpaca import generate_prompt
from scripts.prompt_generate import *
import re
import random
from datasets import load_dataset

dataset = load_dataset("derek-thomas/ScienceQA")
testset = dataset['test']
print(f"test has {len(testset):,} samples")

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Question:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),

    "prompt_QA": (
        "### Question: {q}\n### Context: {context}\n### Choices: {choice}\n### Answer:"
    )
}

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = 1
    # int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    adapter_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    adapter_checkpoint = torch.load(adapter_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    model_args.adapter_layer = int(adapter_checkpoint['adapter_query.weight'].shape[0] / model_args.adapter_len)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    model.load_state_dict(adapter_checkpoint, strict=False)
    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    adapter_path: str,
    temperature: float = 0.1,
    top_p: float = 0.75,
    max_seq_len: int = 540,
    max_batch_size: int = 64,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, adapter_path, local_rank, world_size, max_seq_len, max_batch_size
    )
    # instructs = [
    #     "Tell me about alpacas.",
    #     "Tell me about the president of Mexico in 2019.",
    #     "Tell me about the king of France in 2019.",
    #     "List all Canadian provinces in alphabetical order.",
    #     "Write a Python program that prints the first 10 Fibonacci numbers.",
    #     "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
    #     "Tell me five words that rhyme with 'shock'.",
    #     "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
    #     "Count up from 1 to 500."]
    # prompts = [PROMPT_DICT['prompt_no_input'].format_map({'instruction':x, 'input': ''}) for x in instructs]
    # with open('./ScienceQA_test_text/test.json', encoding='utf-8') as f:
        # data = json.load(f)             
    # https://github.com/ZrrSkywalker/LLaMA-Adapter/issues/16
        # multiple gpu for inference is not possible at this moment with this code!
    # prompts = [PROMPT_DICT['prompt_QA'].format_map({'q':x['question'], 'context': x['hint'], 'choice': x['choices']}) for i, x in data.items()]
    # print(prompts[10])
    
    pattern = re.compile(r'The answer is ([A-Z]).')
    options=['A', 'B', 'C', 'D', 'E']
    cnt = 0
    save_results_json = {}
    all_outputs = []
    all_sbjs = []
    all_ans = []
    prompts = []
    choices = []
    answers = []
    batch = 64
    n_samples = len(testset)
    # indices = np.arange(n_samples)
    # np.random.shuffle(indices)

  
    for i in range(len(testset)):
        prompt = build_prompt(testset[i], test=True)
        prompts.append(prompt)
        choice = testset[i]['choices']
        choices.append(choice)
        answer = testset[i]['answer']
        answers.append(answer)
        subject = testset[i]['subject']
        subjects.append(subject)

    for start in range(0, n_samples, batch):
        end = min(start + batch, n_samples)
        # batch_idx = indices[start:end]
        prompt = prompts[start:end]
        choice = choices[start:end]
        answer = answers[start:end]

        # prompt = prompts[batch_idx]

        results = generator.generate(
                prompt, max_gen_len=540, temperature=temperature, top_p=top_p
            )

        for i in range(len(results)):
            res = pattern.findall(results[i])

            if len(res) == 1:
                pred = res[0]
            else:
                pred = "FAILED"
            
            pred_idx = get_pred_idx(pred, choice[i], options)

            if pred_idx == answer[i]:
                cnt += 1
                print(str(cnt) + ' out of ' + str(end*i))

            
            all_outputs.append(results[i])
            all_ans.append(answer[i])

    acc = (cnt / len(testset)) * 100

    print(acc)

    save_results_json['output'] = all_outputs
    save_results_json['answer'] = all_ans
    save_results_json['subject'] = subjects
    save_results_json['acc'] = acc          

    with open('result.json', 'w') as fp:
        json.dump(save_results_json, fp, indent=4)


if __name__ == "__main__":
    fire.Fire(main)
