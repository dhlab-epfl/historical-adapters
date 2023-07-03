from utils_metrics import get_entities_bio, f1_score, classification_report
from transformers import (LlamaConfig, LlamaModel, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, \
                          BartForConditionalGeneration, BartTokenizer, BartConfig)
import torch
import time
import math
from datasets import load_dataset

# from llama import LLaMA, ModelArgs, Tokenizer, Transformer
import os
from typing import Tuple

from fairscale.nn.model_parallel.initialize import initialize_model_parallel


class InputExample():
    def __init__(self, words, labels):
        self.words = words
        self.labels = labels


def template_entity(words, input_TXT, start):
    # input text -> template
    words_length = len(words)
    words_length_list = [len(i) for i in words]

    window_size = 5

    # word 마다 각 template 찍어낼 리스트 준비
    # input_TXT = [input_TXT] * (window_size * words_length)
    #
    # input_ids = tokenizer(input_TXT, return_tensors='pt')[
    #     'input_ids']  # tokenized input
    model.to(device)

    template_list = [
        " is a location entity .",
        " is a person entity .",
        " is an organization entity .",
        " is an other entity .",
        " is not a named entity ."]

    entity_dict = {0: 'LOC', 1: 'PER', 2: 'ORG', 3: 'MISC', 4: 'O'}

#     words = ['SOCCER', 'SOCCER -', 'SOCCER - JAPAN', 'SOCCER - JAPAN GET', 'SOCCER - JAPAN GET LUCKY',
#     'SOCCER - JAPAN GET LUCKY WIN', 'SOCCER - JAPAN GET LUCKY WIN ,', 'SOCCER - JAPAN GET LUCKY WIN , CHINA']

    temp_list = []
    for i in range(words_length):
        for j in range(len(template_list)):
            temp_list.append(input_TXT + ' ' + words[i] + template_list[j])
            # 각 word 마다 template 찍어냄
            # added input_TXT to the template

#     temp_list = ['SOCCER is a location entity .', 'SOCCER is a person entity .', 'SOCCER is an organization entity .',
#     'SOCCER is an other entity .', 'SOCCER is not a named entity .', 'SOCCER - is a location entity .',
#     'SOCCER - is a person entity .', 'SOCCER - is an organization entity .', 'SOCCER - is an other entity .',
#     'SOCCER - is not a named entity .', 'SOCCER - JAPAN is a location entity .', 'SOCCER - JAPAN is a person entity.',
#     'SOCCER - JAPAN is an organization entity .', ...

    output_ids = tokenizer(temp_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
    # output_ids[:, 0] = 32000 #2  2 is </s> for BART and 0 is <s> --> for LLama it's 32000 [PAD]
    output_length_list = [0] * window_size * words_length

    for i in range(len(temp_list) // window_size):
        base_length = (len(tokenizer(
            temp_list[i * window_size], padding=True, truncation=True)['input_ids'])) - 4
        # base_length = ((tokenizer(temp_list[i*5], padding=True, truncation=True)).shape)[1] - 4
        # print(base_length)
        output_length_list[i * window_size:i * window_size +
                           window_size] = [base_length] * window_size
        output_length_list[i * window_size + 4] += 1

    score = [1] * window_size * words_length
    with torch.no_grad():

        # output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids[:, :output_ids.shape[1] - 2].to(device))[0]

        output = model(output_ids.to(device))[0]
        for i in range(output_ids.shape[1] - 3): # I dont know why its - 3 here
            # print(input_ids.shape)
            # print(i)
            #
            # tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            # "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
            try:
                if i < output.shape[1]:
                    logits = output[:, i, :]
                else:
                    logits = output[:, output.shape[1]-1, :]
            except:
                import pdb;
                pdb.set_trace()
            logits = logits.softmax(dim=1)
            # values, predictions = logits.topk(1,dim = 1)
            logits = logits.to('cpu').numpy()
            # print(output_ids[:, i+1].item())
            for j in range(0, window_size * words_length):
                if i < output_length_list[j]:
                    # print(logits[j].shape)
                    score[j] = score[j] * logits[j][int(output_ids[j][i + 1])]

    end = start + (score.index(max(score)) // 5)
    # score_list.append(score)
    return [start, end, entity_dict[(score.index(max(score)) % window_size)], max(
        score)]  # [start_index,end_index,label,score]


def prediction(input_TXT):
    input_TXT_list = input_TXT.split(' ')

    entity_list = []
    for i in range(len(input_TXT_list)):
        words = []
        for j in range(1, min(9, len(input_TXT_list) - i + 1)):
            word = (' ').join(input_TXT_list[i:i + j])
            words.append(word)

        # [start_index,end_index,label,score]
        entity = template_entity(words, input_TXT, i)
        if entity[1] >= len(input_TXT_list):
            entity[1] = len(input_TXT_list) - 1
        if entity[2] != 'O':
            entity_list.append(entity)
    i = 0
    if len(entity_list) > 1:
        while i < len(entity_list):
            j = i + 1
            while j < len(entity_list):
                if (entity_list[i][1] < entity_list[j][0]) or (
                        entity_list[i][0] > entity_list[j][1]):
                    j += 1
                else:
                    if entity_list[i][3] < entity_list[j][3]:
                        entity_list[i], entity_list[j] = entity_list[j], entity_list[i]
                        entity_list.pop(j)
                    else:
                        entity_list.pop(j)
            i += 1
    label_list = ['O'] * len(input_TXT_list)

    for entity in entity_list:
        label_list[entity[0]:entity[1] + 1] = ["I-" + \
            entity[2]] * (entity[1] - entity[0] + 1)
        label_list[entity[0]] = "B-" + entity[2]
    return label_list


def cal_time(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


# def load(
#     ckpt_dir: str,
#     tokenizer_path: str,
#     adapter_path: str,
#     local_rank: int,
#     world_size: int,
#     max_seq_len: int,
#     max_batch_size: int,
#     quantizer: bool=False,
# ) -> LLaMA:
#     start_time = time.time()
#     checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
#     assert world_size == len(
#         checkpoints
#     ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
#     ckpt_path = checkpoints[local_rank]
#     print("Loading")
#     checkpoint = torch.load(ckpt_path, map_location="cpu")
#     adapter_checkpoint = torch.load(adapter_path, map_location="cpu")
#     with open(Path(ckpt_dir) / "params.json", "r") as f:
#         params = json.loads(f.read())
#
#     model_args: ModelArgs = ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
#     model_args.adapter_layer = int(adapter_checkpoint["adapter_query.weight"].shape[0] / model_args.adapter_len)
#     tokenizer = Tokenizer(model_path=tokenizer_path)
#     model_args.vocab_size = tokenizer.n_words
#     torch.set_default_tensor_type(torch.cuda.HalfTensor)
#     model = Transformer(model_args)
#     print(model)
#     torch.set_default_tensor_type(torch.FloatTensor)
#     model.load_state_dict(checkpoint, strict=False)
#     model.load_state_dict(adapter_checkpoint, strict=False)
#     generator = LLaMA(model, tokenizer)
#     print(f"Loaded in {time.time() - start_time:.2f} seconds")
#     return model

# local_rank, world_size = setup_model_parallel()
# if local_rank > 0:
#     sys.stdout = open(os.devnull, "w")

tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf', pad_token='[PAD]')

# input_TXT = "Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C
# championship match on Friday ."f.token_type_ids for f in features] model = load('/data1/data/sooh-data/llama/7B',
# '/data1/data/sooh-data/llama/tokenizer.model',
# '/data1/data/sooh-data/llama/hipe/checkpoint/adapter_adapter_len10_layer30_epoch9.pth', local_rank, world_size,
# 512, 32, False) model = BartForConditionalGeneration.from_pretrained('facebook/bart-large') model =
# BartForConditionalGeneration.from_pretrained('../dialogue/bart-large')
model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
model.eval()
# input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']
# print(input_ids)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

score_list = []
dataset = load_dataset("conll2003")
test_dataset = dataset['test']
guid_index = 1
examples = []
words = []
labels = []
ner_dict = {
    'O': 0,
    'B-PER': 1,
    'I-PER': 2,
    'B-ORG': 3,
    'I-ORG': 4,
    'B-LOC': 5,
    'I-LOC': 6,
    'B-MISC': 7,
    'I-MISC': 8}
ner_inv_dict = {v: k for k, v in ner_dict.items()}

for i in range(len(test_dataset)):
    line = test_dataset[i]
    words = line['tokens']
    labels = line['ner_tags']
    tagged_labels = []
    for j in range(len(labels)):
        tagged_labels.append(ner_inv_dict[labels[j]])

    examples.append(InputExample(words=words, labels=tagged_labels))
trues_list = []
preds_list = []
str = ' '
num_01 = len(examples)
num_point = 0
start = time.time()
from tqdm import tqdm
for example in tqdm(examples, total=len(examples)):
    sources = str.join(example.words)
    preds_list.append(prediction(sources))
    trues_list.append(example.labels)
    print('%d/%d (%s)' % (num_point + 1, num_01, cal_time(start)))
    print('Pred:', preds_list[num_point])
    print('Gold:', trues_list[num_point])
    num_point += 1


true_entities = get_entities_bio(trues_list)
pred_entities = get_entities_bio(preds_list)
results = {
    "f1": f1_score(true_entities, pred_entities)
}
print(results["f1"])
for num_point in range(len(preds_list)):
    preds_list[num_point] = ' '.join(preds_list[num_point]) + '\n'
    trues_list[num_point] = ' '.join(trues_list[num_point]) + '\n'
with open('./pred.txt', 'w') as f0:
    f0.writelines(preds_list)
with open('./gold.txt', 'w') as f0:
    f0.writelines(trues_list)
