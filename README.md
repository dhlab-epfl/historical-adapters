# historical-adapters
The repository for the development of a LLaMA adapter for historical documents.

## Fine-tuning

### Org
Please refer:
https://github.com/ZrrSkywalker/LLaMA-Adapter/tree/main

### lit
In this implementation, I referred lit-directory for fine-tuning.
Please refer (here)[https://github.com/Lightning-AI/lit-llama] for details!

1. Clone the lit git
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
[Note] If you are using 8 gpus, you can fine-tune the model under 1 hour! (I used 4 gpus, and it took about 6 hours 🤔)

## Inference

### lit-adapter generation
https://github.com/Lightning-AI/lit-llama/blob/main/generate_adapter.py

#### ScienceQA

1. Used LLaMA 7B model from [Meta](https://github.com/facebookresearch/llama)
2. Run 
```
python generate_adapter.py
```
-- You need lit format converted weights of fine-tuned LLaMA with alpaca dataset on adapter technique






