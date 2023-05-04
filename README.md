# historical-adapters
The repository for the development of a LLaMA adapter for historical documents.

## Inference

1. Download LLaMA 7B from [huggingface](https://huggingface.co/nyanko7/LLaMA-7B)
2. Run `python ./alpaca_finetuning_v1/extract_adapter_from_checkpoint.py` (You need weights of fine-tuned LLaMA with alpaca dataset on adapter technique)
3. Run `bash generate.sh`
