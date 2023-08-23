# historical-adapters
The repository for the experiments to validate the efficiency of using [LLaMA 7B-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter) on the historical dataset compared to in-context learning (i.e., zero- and three-shot learnings). 

## Dataset

We conduct experiments with Closed-domain QA (i.e., ScienceQA), Open-domain QA (i.e., ArchivalQA), and Named Entity Recognition (i.e., HIPE) tasks. You can find the all dataset in `data` folder. The original git repository for each dataset is as follow:

* [ScienceQA](https://github.com/lupantech/ScienceQA)
* [ArchivalQA](https://github.com/WangJiexin/ArchivalQA/tree/main)
* [HIPE](https://github.com/hipe-eval/HIPE-2022-data/tree/main)

## Fine-tuning 

We follow the fine-tuning process of [LLaMA-Adapter V1](https://github.com/OpenGVLab/LLaMA-Adapter/tree/main/alpaca_finetuning_v1) with the alpaca dataset. We use text-only modality and used LLaMA 7B model.

### HIPE dataset for NER task



### ScienceQA dataset for Closed-domain QA task

### ArchivalQA dataset for Open-domain QA task

### Fine-tuning with ScienceQA dataset

## In-Context learning

### Zero- and Three-shot learnings 





