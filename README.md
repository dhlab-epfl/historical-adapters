# historical-adapters
The repository for the development of a LLaMA adapter for historical documents.

## Fine-tuning

### Fine-tuning with alpaca

#### Original Repo
Please refer [here](https://github.com/ZrrSkywalker/LLaMA-Adapter/tree/main)

#### Lightning Repo
In this implementation, I referred lit-directory for fine-tuning: [Pytorch Lightning framework](https://lightning.ai/docs/pytorch/stable/)
Please refer [here](https://github.com/Lightning-AI/lit-llama) for details!

1. Clone the git
```
git clone https://github.com/Lightning-AI/lit-llama
cd lit-llama
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Convert the weights into lit format
```
python scripts/convert_checkpoint.py --model_size 7B
```
4. Fine tuning LLaMA with adapter technique using alpaca dataset
```
python finetune_adapter.py
```
[Note] If you are using 8 gpus, you can fine-tune the model under 1 hour

### Fine-tuning with ScienceQA dataset

#### Original repo framework fine-tuning

```
bash alpaca_finetuning_v1/finetuning.sh
```

#### Lightning framework fine-tuning
https://github.com/Lightning-AI/lit-llama/blob/main/howto/finetune_adapter.md

1. Prepare customized ScienceQA dataset

```
python scripts/prepare_scienceQA.py
```

2. Fine-tuning

```
python finetune-adapter.py
```

## Inference

### lit-adapter generation

https://github.com/Lightning-AI/lit-llama/blob/main/generate_adapter.py

#### ScienceQA

[Note] You need lit format converted weights of fine-tuned LLaMA with alpaca dataset on adapter technique.

1. Install dependencies
```
pip install -r requirements.txt
```

2. Run 
```
python generate_adapter.py
```




