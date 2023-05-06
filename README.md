# historical-adapters
The repository for the development of a LLaMA adapter for historical documents.

## Fine-tuning

1. Build conda env `conda create -n <env_name> -y python=3.8`
2. Install dependency
```
pip install -r requirements.txt
pip install -e .
```
3. Run script as below. Please change the `TARGET_FOLDER` and `output_dir` accordingly

`bash run.sh`


## Inference

1. Used LLaMA 7B model from [Meta](https://github.com/facebookresearch/llama)
2. Run `python ./alpaca_finetuning_v1/extract_adapter_from_checkpoint.py` (You need weights of fine-tuned LLaMA with alpaca dataset on adapter technique)
3. Run `bash generate.sh`

### lit-adapter generation

https://github.com/Lightning-AI/lit-llama/blob/main/generate_adapter.py





