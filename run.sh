


torchrun --nproc_per_node 1 example.py \
         --ckpt_dir "/nlpdata1/home/sooh/llama/7B" \
         --tokenizer_path "/nlpdata1/home/sooh/llama/tokenizer.model" \
         --adapter_path "/nlpdata1/home/sooh/llama_adapter/adapter_adapter_len10_layer30_epoch5.pth"
         