import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch
import json

from generate import generate
from lit_llama import Tokenizer
from lit_llama.adapter import LLaMA
from lit_llama.utils import EmptyInitOnDevice, lazy_load, llama_model_lookup
from scripts.prepare_alpaca import generate_prompt

PROMPT_DICT = {
    "prompt_QA": (
        # "Choose the correct answer from Choices."
        "\n### Question: {q}\n### Context: {context}\n### Choices: {choice}\n### Answer:"
    )
}

numbering = {
    'scienceQA': (
        "({i}) {ele}"
    )
}

def main(
    prompt: str = "What food do lamas eat?",
    input: str = "",
    adapter_path: Optional[Path] = None,
    pretrained_path: Optional[Path] = None,
    tokenizer_path: Optional[Path] = None,
    quantize: Optional[str] = None,
    max_new_tokens: int = 50,
    top_k: int = 200,
    temperature: float = 0.8,
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned LLaMA-Adapter model.
    See `finetune_adapter.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        adapter_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune_adapter.py`.
        input: Optional input (Alpaca style).
        pretrained_path: The path to the checkpoint with pretrained LLaMA weights.
        tokenizer_path: The tokenizer path to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
    """
    if not adapter_path:
        adapter_path = Path("/nlpdata1/home/sooh/lit-llama/adapter/lit-llama-adapter-finetuned.pth")
    if not pretrained_path:
        pretrained_path = Path(f"/nlpdata1/home/sooh/lit-llama/7B/lit-llama.pth")
    if not tokenizer_path:
        tokenizer_path = Path("/nlpdata1/home/sooh/lit-llama/tokenizer.model")
    
    assert adapter_path.is_file()
    assert pretrained_path.is_file()
    assert tokenizer_path.is_file()

    fabric = L.Fabric(devices=1)
    # fabric.launch()
    dtype = torch.bfloat16 if fabric.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    # pretrained_checkpoint = lazy_load(pretrained_path)
    # adapter_checkpoint = lazy_load(adapter_path)
    # with (lazy_load(pretrained_path) as pretrained_checkpoint,lazy_load(adapter_path) as adapter_checkpoint):
    with lazy_load(pretrained_path) as pretrained_checkpoint:
        with lazy_load(adapter_path) as adapter_checkpoint:
        # adapter_checkpoint = lazy_load(adapter_path)
            name = llama_model_lookup(pretrained_checkpoint)

            with EmptyInitOnDevice(
                device=fabric.device, dtype=dtype, quantization_mode=quantize
            ):
                model = LLaMA.from_name(name)

            # 1. Load the pretrained weights
            model.load_state_dict(pretrained_checkpoint, strict=False)
            # 2. Load the fine-tuned adapter weights
            model.load_state_dict(adapter_checkpoint, strict=False)

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup_module(model)

    tokenizer = Tokenizer(tokenizer_path)

    data_idx_list = []
    with open('./ScienceQA_test_text/test.json', encoding='utf-8') as f:
        data = json.load(f)    
        for i, ele in data.items():
            data_idx_list.append(i)


    # prompts = [PROMPT_DICT['prompt_QA'].format_map({'q':x['question'], 'context': x['hint'], 'choice': x['choices']}) for i, x in data.items()]
    num_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
    prompts = [PROMPT_DICT['prompt_QA'].format_map({'q':x['question'], 'context': x['hint'], 'choice': ' '.join([numbering['scienceQA'].format_map({'i': num_dict[idx], 'ele': ele}) for idx, ele in enumerate(x['choices'])])}) for i, x in data.items()]
    # sample = {"instruction": prompt, "input": input}
    # prompt = generate_prompt(sample)
    
    save_results_json = {}
    all_prompts = []
    all_outputs = []
    all_ans = []

    acc = 0
    for idx in range(len(prompts)):
        encoded = tokenizer.encode(prompts[idx], bos=True, eos=False, device=model.device)

        t0 = time.perf_counter()
        output = generate(
            model,
            idx=encoded,
            max_seq_length=max_new_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=tokenizer.eos_id
        )
        t = time.perf_counter() - t0

        output = tokenizer.decode(output)
        output = output.split("### Answer:")[1].strip()

        print(prompts[idx])
        print('================Prediction================')
        print(output)
        print('================Correct Answer================')
        gt = data[data_idx_list[idx]]['choices'][data[data_idx_list[idx]]['answer']]
        print(gt)
    
        if num_dict[data[data_idx_list[idx]]['answer']] in output:
            acc += 1
            print('Correct!')
        elif gt in output:
            acc += 1
            print('Correct!')

        all_prompts.append(prompts[idx])
        all_outputs.append(output)
        all_ans.append(gt)
            

    print('*****************Accuracy*****************')
    print((acc / len(prompts)) * 100)

    save_results_json['prompts'] = all_prompts
    save_results_json['output'] = all_outputs
    save_results_json['answer'] = all_ans
    save_results_json['acc'] = (acc / len(prompts)) * 100

    with open('result.json', 'w') as fp:
        json.dump(save_results_json, fp, indent=4)
        # print(f"\n\nTime for inference: {t:.02f} sec total, {max_new_tokens / t:.02f} tokens/sec", file=sys.stderr)
        # if fabric.device.type == "cuda":
        #     print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(main)
