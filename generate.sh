torchrun --nproc_per_node 1 example-multiple-archiv.py \
--ckpt_dir /data1/data/sooh-data/llama/7B/ \
--tokenizer_path /data1/data/sooh-data/llama/tokenizer.model \
--adapter_path /data1/data/sooh-data/llama/archiv/checkpoint2/adapter_adapter_len10_layer30_epoch24.pth
