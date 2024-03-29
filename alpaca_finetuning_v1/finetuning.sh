TARGET_FOLDER='/data1/data/sooh-data/llama'

torchrun --nproc_per_node 7 finetuning_hipe_prompt2.py \
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
    --output_dir /data1/data/sooh-data/llama/hipe/checkpoint-prompt2/
