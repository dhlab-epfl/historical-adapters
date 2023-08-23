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
import pandas as pd
from inference.eval_em_f1 import *

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



PROMPT_DICT = {

    "ArchivalQA": (
        "###Question:\n{question}\n\n### Answer:"

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
    test_dataset = pd.read_csv('./data/ArchivalQA/data/ArchivalQA_test.csv')

    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, adapter_path, local_rank, world_size, max_seq_len, max_batch_size
    )
    
    em_score = []
    f1_score = []
    save_results_json = {}
    results = {}
    all_outputs = []
    all_preds = []
    questions = []
    answers = []
    batch = 64
    n_samples = len(test_dataset)

  
    for i in range(len(test_dataset)):
        question = test_dataset.iloc[i]['question']
        questions.append(question)

        answer = test_dataset.iloc[i]['answer']
        answers.append(answer)

    for start in range(0, n_samples, batch):
        end = min(start + batch, n_samples)
        # batch_idx = indices[start:end]
        prompt = questions[start:end]
        answer = answers[start:end]

        # prompt = prompts[batch_idx]
        prompt = [PROMPT_DICT["ArchivalQA"].format_map({"question": x}) for x in prompt]
        # print(prompt[0])
        results = generator.generate(
                prompt, max_gen_len=540, temperature=temperature, top_p=top_p
            )
        
        for i in range(len(results)):
            
            pred = results[i].split('Answer:')[1]
            
            em_score.append(compute_em(answer[i], pred))
            f1_score.append(compute_f1(answer[i], pred))
    
            all_outputs.append(results[i])
            all_preds.append(pred)

            print('====Scores====')
            print(100*sum(em_score)/len(em_score))
            print(100*sum(f1_score)/len(f1_score))
            print('========')
    
    assert len(em_score) == len(test_dataset)
    assert len(f1_score) == len(test_dataset)

    em = 100.0 * sum(em_score) / len(test_dataset)
    f1 = 100.0 * sum(f1_score) / len(test_dataset)
   
    print(em)
    print(f1)

    save_results_json['EM'] = em    
    save_results_json['F1'] = f1   
    save_results_json['output'] = all_outputs
    save_results_json['prediction'] = all_preds   

    with open('./results/org-finetuned-archiv-result-latest.json', 'w') as fp:     
        json.dump(save_results_json, fp, indent=4)


if __name__ == "__main__":
    fire.Fire(main)


