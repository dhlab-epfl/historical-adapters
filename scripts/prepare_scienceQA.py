"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import sys
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
import requests
import json
from torch.utils.data import random_split
from lit_llama.tokenizer import Tokenizer
from tqdm import tqdm
from datasets import load_dataset
from scripts.prompt_generate import *

dataset = load_dataset("derek-thomas/ScienceQA")

# DATA_FILE = "https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned_archive.json"
# DATA_FILE_NAME = "alpaca_data_cleaned_archive.json"
IGNORE_INDEX = -1


def prepare(
    # destination_path: Path = Path("data/alpaca"), 
    tokenizer_path: Path = Path("/nlpdata1/home/sooh/lit-llama/tokenizer.model"),
    test_split_size: int = 4241,
    max_seq_length: int = 256,
    seed: int = 42,
    mask_inputs: bool = False,  # as in alpaca-lora
    # data_file_name: str = DATA_FILE_NAME
) -> None:
    """Prepare the Alpaca dataset for instruction tuning.
    
    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    
    # destination_path.mkdir(parents=True, exist_ok=True)
    # file_path = destination_path / data_file_name
    # download(file_path)

    # TODO: If we don't have the Meta weights, where do we get the tokenizer from?
    tokenizer = Tokenizer(tokenizer_path)
    
    # with open(file_path, "r") as file:
    #     data = json.load(file)


    # # Partition the dataset into train and test
    # train_split_size = len(data) - test_split_size
    # train_set, test_set = random_split(
    #     data, 
    #     lengths=(train_split_size, test_split_size),
    #     generator=torch.Generator().manual_seed(seed),
    # )
    # train_set, test_set = list(train_set), list(test_set)
    train_set, test_set = dataset['train'], dataset['validation']
    print(f"train has {len(train_set):,} samples")
    print(f"val has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(train_set)]
    torch.save(train_set, "/nlpdata1/home/sooh/lit-llama/science/train.pt")

    print("Processing val split ...")
    test_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(test_set)]
    torch.save(test_set, "/nlpdata1/home/sooh/lit-llama/science/val.pt")


def download(file_path: Path):
    """Downloads the raw json data file and saves it in the given destination."""
    if file_path.exists():
        return
    with open(file_path, "w") as f:
        f.write(requests.get(DATA_FILE).text)


def prepare_sample(example: dict, tokenizer: Tokenizer, max_length: int, mask_inputs: bool = True):
    """Processes a single sample.
    
    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The prompt text is formed as a single message including both
    the instruction and the input. The label/target is the same message but with the
    response attached.

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    full_prompt = build_prompt(example, test=True)
    full_prompt_and_response = build_prompt(example, test=False)
    encoded_full_prompt = tokenize(tokenizer, full_prompt, max_length=max_length, eos=False)
    encoded_full_prompt_and_response = tokenize(tokenizer, full_prompt_and_response, eos=True, max_length=max_length)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[:len(encoded_full_prompt)] = IGNORE_INDEX

    return {**example, "input_ids": encoded_full_prompt_and_response, "input_ids_no_response": encoded_full_prompt, "labels": labels}


def tokenize(tokenizer: Tokenizer, string: str, max_length: int, eos=True) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos, max_length=max_length)

    

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
