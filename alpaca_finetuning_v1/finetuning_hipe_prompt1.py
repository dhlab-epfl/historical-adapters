import os
import argparse
import datetime
import json
import time
import copy
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import pickle

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_finetuning import train_one_epoch, val_one_epoch
from transformers import BertTokenizer, GPT2Tokenizer
from llama import ModelArgs, Transformer, Tokenizer, LLaMA
import models_llama_adapter
from datasets import load_dataset
from util.prompt_generate import *
import wandb


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "ArchivalQA": (
        "###Question:\n{question}\n\n### Answer:"

        ),

    "HIPE": (
    
        "INPUT: {context}\n\nOUTPUT: "
    )

}

class InputExample():
    def __init__(self, words, labels, nat_labels):
        self.words = words
        self.labels = labels
        self.nat_labels = nat_labels # Ground truth

template_list = [" is a time entity.", " is a location entity.", " is a person entity.", " is an organization entity.",
                " is an product entity.", " is not a named entity."]
entity_dict = {'TIME':0, 'LOCATION':1, 'PERSON':2, 'ORGANIZATION':3, 'PRODUDCT':4, 'O':5}
inv_entity = {v: k for k, v in entity_dict.items()}

class InstructionDataset(Dataset):
    def __init__(self, model_path, max_words=30, partition='train'):
       
    # ScienceQA dataset loading    
        # dataset = load_dataset("derek-thomas/ScienceQA")
        
        # if partition == 'train':
        #    self.ann = dataset['train']
        # else:
        #    self.ann = dataset['validation']

    # HIPE dataset loading    
        with open(f'../data/HIPE/parag-label-HIPE-train.pickle', 'rb') as file1:
            train_dataset = pickle.load(file1)
        with open(f'../data/HIPE/parag-label-HIPE-dev.pickle', 'rb') as file2:
            val_dataset = pickle.load(file2)

        # HIPE
        if partition == 'train':
            total = []
            total_ans = []
            for i in range(len(train_dataset)):
                temp = {}
                data = train_dataset[i]
                context = ' '.join(data.words)

                for j in data.nat_labels:
                    temp_dict = {}
                    token = j.split(' is')[0]
                    label = ' is' + j.split(' is')[1]
                    entity = inv_entity[template_list.index(label)]
                    if entity != 'O':
                        temp_dict['entity'] = entity
                        temp_dict['text'] = token
           
                        total_ans.append(temp_dict)

                temp['context'] = context
                temp['answer'] = total_ans
                total.append(temp)
            self.ann = total
        else:
            total = []
            for i in range(len(val_dataset)):
                temp = {}
                data = val_dataset[i]
                context = ' '.join(data.words)

                for j in data.nat_labels:
                    temp_dict = {}
                    token = j.split(' is')[0]
                    label = ' is' + j.split(' is')[1]
                    entity = inv_entity[template_list.index(label)]
                    if entity != 'O':
                        temp_dict['entity'] = entity
                        temp_dict['text'] = token
           
                        total_ans.append(temp_dict)

                temp['context'] = context
                temp['answer'] = total_ans
                total.append(temp)
            self.ann = total

        self.max_words = max_words
        tokenizer = Tokenizer(model_path= model_path + './tokenizer.model')
        self.tokenizer1 = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        #print(ann)
    # ScienceQA
        # prompt = build_prompt(ann, test=True)
        # example = build_prompt(ann, test=False)
        
        prompt = PROMPT_DICT['HIPE'].format_map(ann)
        example = prompt + str(ann['answer'])

        prompt = torch.tensor(self.tokenizer1.encode(prompt, bos=True, eos=False), dtype=torch.int64)
        example = torch.tensor(self.tokenizer1.encode(example, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[:self.max_words]
        labels = copy.deepcopy(example)
        labels[:len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return example, labels, example_mask


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--llama_model_path', default='./llama', type=str,
                        help='path of llama model')
    parser.add_argument('--model', default='llama7B_adapter', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--adapter_layer', type=int, default=30, metavar='LENGTH',
                        help='the number of adapter layer')


    parser.add_argument('--adapter_len', type=int, default=10, metavar='LENGTH',
                        help='the adapter length')

    parser.add_argument('--max_seq_len', type=int, default=512, metavar='LENGTH',
                        help='the maximum sequence length')


    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/instruction_dataset/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=8, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # parser.add_argument('--deepspeed', type=str)

    return parser


def main(args):
    wandb.init(
    # set the wandb project where this run will be logged
    project="adapter-hipe-context-prompt1",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": args.lr,
    "architecture": "adapter",
    "dataset": "HIPE",
    "epochs": args.epochs,
    }
)
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = InstructionDataset(model_path = args.llama_model_path, max_words=args.max_seq_len, partition='train')
    dataset_val = InstructionDataset(model_path = args.llama_model_path, max_words=args.max_seq_len, partition='val')
    
    print("=================DATA VALIDATION=================")
    print(dataset_train)
    print(dataset_val)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    
    # define the model
    model = models_llama_adapter.__dict__[args.model](args)
    
    model.to(device)
    wandb.watch(model)
    
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            data_loader_val.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        val_stats = val_one_epoch(
        model, data_loader_val,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir and (epoch % 8 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch, 
                        **{f'val_{k}': v for k, v in val_stats.items()}}
        
        wandb.log(log_stats)


        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
