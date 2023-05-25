torchrun --nproc_per_node 1 example-multiple.py \
--ckpt_dir /data1/data/sooh-data/llama/7B/ \
--tokenizer_path /data1/data/sooh-data/llama/tokenizer.model \
--adapter_path /data1/data/sooh-data/llama/checkpoint2/adapter_adapter_len10_layer30_epoch5.pth
