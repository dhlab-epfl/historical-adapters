# historical-adapters
The repository for the experiments to validate the efficiency of using [LLaMA 7B-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter) on the historical dataset compared to in-context learning (i.e., zero- and three-shot learnings). 

## Dataset

We conduct experiments with Closed-domain QA (i.e., ScienceQA), Open-domain QA (i.e., ArchivalQA), and Named Entity Recognition (i.e., HIPE) tasks. You can find the all dataset in `data` folder. The original git repository for each dataset is as follows:

* [ScienceQA](https://github.com/lupantech/ScienceQA)
* [ArchivalQA](https://github.com/WangJiexin/ArchivalQA/tree/main)
* [HIPE](https://github.com/hipe-eval/HIPE-2022-data/tree/main)

## Fine-tuning 

We follow the fine-tuning process of [LLaMA-Adapter V1](https://github.com/OpenGVLab/LLaMA-Adapter/tree/main/alpaca_finetuning_v1) with the alpaca dataset. We use text-only modality and LLaMA 7B model.

### ScienceQA

Run `bash alpaca_finetuning_v1/finetuning.sh` with the command below:

```
TARGET_FOLDER='/data1/data/sooh-data/llama'

torchrun --nproc_per_node 8 finetuning_science.py \
    --model Llama7B_adapter \
    --llama_model_path $TARGET_FOLDER/ \
    --adapter_layer 30 \
    --adapter_len 10 \
    --max_seq_len 512 \
    --batch_size 4 \
    --epochs 10 \
    --warmup_epochs 2 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --output_dir /data1/data/sooh-data/llama/science/checkpoint/
```
Specify the argument `--nproc_per_node` with the available GPU number of your working environment.


### ArchivalQA

Run `bash alpaca_finetuning_v1/finetuning.sh` with the command below:

```
TARGET_FOLDER='/data1/data/sooh-data/llama'

torchrun --nproc_per_node 8 finetuning_archiv.py \
    --model Llama7B_adapter \
    --llama_model_path $TARGET_FOLDER/ \
    --adapter_layer 30 \
    --adapter_len 10 \
    --max_seq_len 512 \
    --batch_size 4 \
    --epochs 10 \
    --warmup_epochs 2 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --output_dir /data1/data/sooh-data/llama/archiv/checkpoint/
```
Specify the argument `--nproc_per_node` with the available GPU number of your working environment.


### HIPE

#### Prompt 1: JSON format prompt

Run `bash alpaca_finetuning_v1/finetuning.sh` with the command below:

```
TARGET_FOLDER='/data1/data/sooh-data/llama'

torchrun --nproc_per_node 8 finetuning_hipe_prompt1.py \
    --model Llama7B_adapter \
    --llama_model_path $TARGET_FOLDER/ \
    --adapter_layer 30 \
    --adapter_len 10 \
    --max_seq_len 512 \
    --batch_size 4 \
    --epochs 10 \
    --warmup_epochs 2 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --output_dir /data1/data/sooh-data/llama/hipe/checkpoint/
```
Specify the argument `--nproc_per_node` with the available GPU number of your working environment.

#### Prompt 2: ChatGPT prompt

Run `bash alpaca_finetuning_v1/finetuning.sh` with the command below:

```
TARGET_FOLDER='/data1/data/sooh-data/llama'

torchrun --nproc_per_node 8 finetuning_hipe_prompt2.py \
    --model Llama7B_adapter \
    --llama_model_path $TARGET_FOLDER/ \
    --adapter_layer 30 \
    --adapter_len 10 \
    --max_seq_len 512 \
    --batch_size 4 \
    --epochs 10 \
    --warmup_epochs 2 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --output_dir /data1/data/sooh-data/llama/hipe/checkpoint/
```
Specify the argument `--nproc_per_node` with the available GPU number of your working environment.


## Inference

### ScienceQA

Run `bash generate.sh` with the command below:

```
torchrun --nproc_per_node 1 example_science.py \
--ckpt_dir /data1/data/sooh-data/llama/7B \
--tokenizer_path /data1/data/sooh-data/llama/tokenizer.model \
--adapter_path /data1/data/sooh-data/llama/science/checkpoint/adapter_adapter_len10_layer30_epoch9.pth
```

Specify the arguments based on your working environment.

##### Evaluation

Run `inference/eval_acc.py` with the revising the code line 93 based on your result file path:

```
scores = get_scores(scienceQA_generation_result_path, testset)
```

### ArchivalQA

Run `bash generate.sh` with the command below:

```
torchrun --nproc_per_node 1 example_archiv.py \
--ckpt_dir /data1/data/sooh-data/llama/7B \
--tokenizer_path /data1/data/sooh-data/llama/tokenizer.model \
--adapter_path /data1/data/sooh-data/llama/archiv/checkpoint/adapter_adapter_len10_layer30_epoch9.pth
```

Specify the arguments based on your working environment.

##### Evaluation

Evaluation is conducted simultaneously within the generation. You can find the code `inference/eval_em_f1.py`.

### HIPE

#### Prompt 1: JSON format prompt

Run `bash generate.sh` with the command below:

```
torchrun --nproc_per_node 1 example_hipe_p1.py \
--ckpt_dir /data1/data/sooh-data/llama/7B \
--tokenizer_path /data1/data/sooh-data/llama/tokenizer.model \
--adapter_path /data1/data/sooh-data/llama/hipe/checkpoint/adapter_adapter_len10_layer30_epoch9.pth
```

Specify the arguments based on your working environment.

##### Evaluation

Run `inference/evaluate_hipe_p1.py` with revising code line 203 based on your result file path:

```
filename = 'results/org-finetuned-hipe-prompt1-result.json'
```


#### Prompt 2: ChatGPT prompt

Run `bash generate.sh` with the command below:

```
torchrun --nproc_per_node 1 example_hipe_p2.py \
--ckpt_dir /data1/data/sooh-data/llama/7B \
--tokenizer_path /data1/data/sooh-data/llama/tokenizer.model \
--adapter_path /data1/data/sooh-data/llama/hipe/checkpoint2/adapter_adapter_len10_layer30_epoch9.pth
```

Specify the arguments based on your working environment.

##### Evaluation

Run `inference/evaluate_hipe_p2_v2.py` with revising code line 29 based on your result file path:

```
filename = 'results/org-finetuned-hipe-prompt2-result.json'
```
