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
import pickle
import pandas as pd

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA
import pdb

from scripts.prepare_alpaca import generate_prompt
from scripts.prompt_generate import *
import re
import random
from datasets import load_dataset
from eval_acc import *

class InputExample():
    def __init__(self, words, labels, nat_labels, all_entity):
        self.words = words
        self.labels = labels
        self.nat_labels = nat_labels # Ground truth
        self.all_entity = all_entity


PROMPT_DICT = {

    "ArchivalQA": (
        "###Question:\n{question}\n\n### Answer:"

        ),

    "HIPE": (
    
        """
        You are working as a named entity recognition expert and your task is to label a given text with named entity labels.
        Your task is to identify and label any named entities present in the text. 
        The named entity labels that you will be using are TIME (time), LOC (location), PER (person), ORG (organization), and PROD (product).
        You may encounter multi-word entities, to make sure to label each word of the entity with the appropriate perfix ("B" for the first word of the entity, "I" for any non-initial word of the entity).
        For words which are not part of any named entity, you should return "O".

        NOTE: Your output format should be a list of tuples, where each tuple consists of a word from the input text and its corresponding named entity label.

        Input: {input}


        Output: 
        """
    )
}

class InputExample():
    def __init__(self, words, labels, nat_labels):
        self.words = words
        self.labels = labels
        self.nat_labels = nat_labels # Ground truth


def split(strng, sep, pos):
    strng = strng.split(sep)
    return sep.join(strng[:pos]), sep.join(strng[pos:])

def setup_model_parallel() -> Tuple[int, int]:
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    print(rank, local_rank, world_size)

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size
    # return rank, world_size

def load(
    ckpt_dir: str,
    tokenizer_path: str,
    adapter_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: 2500,
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
    max_seq_len: int = 2500,
    max_batch_size: int = 2,
):
    # test_dataset = json.load(open('./data/HIPE/HIPE_converted_test_fr.json'))
    

    with open(f'./data/HIPE/parag-label-HIPE-test.pickle', 'rb') as file:
        examples = pickle.load(file)

    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, adapter_path, local_rank, world_size, max_seq_len, max_batch_size
    )
    
    save_results_json = {}
    results = {}
    all_outputs = []
    all_preds = []
    hypothesis = []
    answers = []
    batch = 1
    cnt = 0
    n_samples = len(examples)

    total = []
    total_ans = []
    for i in range(len(examples)):
        data = examples[i]
        input = data.words
        temp = {}
        total_ans = []

        for j in range(len(data.labels)):
            label = data.labels[j].upper()
            total_ans.append((input[j], label))
   

        hypothesis.append(input)
        answers.append(total_ans)

    for start in range(0, len(hypothesis), batch):
        end = min(start + batch, len(hypothesis))
        # batch_idx = indices[start:end]
        prompt = hypothesis[start:end]
        answer = answers[start:end]

        # prompt = prompts[batch_idx]
        prompt = [PROMPT_DICT["HIPE"].format_map({"input": x}) for x in prompt]
        
        print(len(prompt[0].split(' ')))
        if len(prompt[0].split(' ')) < 1000:

            results = generator.generate(
                    prompt, max_gen_len=256, temperature=temperature, top_p=top_p
                )
            print(results)
        else:
            results = ['']

        for i in range(len(results)):
            all_outputs.append(results[i])

    save_results_json['output'] = all_outputs


    with open('./results/org-finetuned-hipe-prompt2-result.json', 'w') as fp:     
        json.dump(save_results_json, fp, indent=4)


if __name__ == "__main__":
    fire.Fire(main)
