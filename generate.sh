torchrun --nproc_per_node 1 example.py \
--ckpt_dir ./alpaca_finetuning_v1/LLaMA-7B/ \
--tokenizer_path ./alpaca_finetuning_v1/LLaMA-7B/tokenizer.model \
--adapter_path ./alpaca_finetuning_v1/adapter_adapter_len10_layer30_epoch5.pth
