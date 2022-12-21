from dataloader import load_file
from transformers import AutoModelForMaskedLM,AutoTokenizer
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
import torch.nn as nn
import random
import argparse
import sys
from dataloader import SST2Dataset, SNLIDataset
from models.modeling import PTuneForClassification 

def construct_generation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default='SST-2')
    parser.add_argument("--data_path", type=str, default='./data')
    parser.add_argument("--PLM_name", type=str, default='bert-base-cased')
    parser.add_argument("--print_num", type=int, default=50)
    parser.add_argument("--eval_num", type=int, default=200)
    parser.add_argument("--quick_exp_data_num", type=int, default=10000)  
    parser.add_argument("--epoch", type=int, default=50)  

    args = parser.parse_args()
    return args
args = construct_generation_args()

def set_seed(seed=34):
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed()

model_name = args.PLM_name
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
pretrained_model = AutoModelForMaskedLM.from_pretrained(model_name)

task_name = args.task_name
data_path = args.data_path

if args.task_name == 'SST-2':
    train_texts, train_labels = load_file(data_path,task_name,'train')
    dev_texts, dev_labels = load_file(data_path,task_name,'dev')
    if args.quick_exp_data_num is not None:
        train_texts, train_labels = train_texts[:args.quick_exp_data_num], train_labels[:args.quick_exp_data_num]
    train_set = SST2Dataset(args, train_texts, train_labels)
    dev_set = SST2Dataset(args, dev_texts, dev_labels)
elif args.task_name == 'SNLI':
    train_texts_a, train_texts_b, train_labels = load_file(data_path,task_name,'train')
    dev_texts_a, dev_texts_b, dev_labels = load_file(data_path,task_name,'dev')
    if args.quick_exp_data_num is not None: 
        train_texts_a, train_texts_b, train_labels = train_texts_a[:args.quick_exp_data_num], train_texts_b[:args.quick_exp_data_num], train_labels[:args.quick_exp_data_num]
    train_set = SNLIDataset(args, train_texts_a, train_texts_b, train_labels)
    dev_set = SNLIDataset(args, dev_texts_a, dev_texts_b, dev_labels)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True)
dev_loader = DataLoader(dev_set, batch_size=32, shuffle=False, drop_last=False)
 

def train():
    model = PTuneForClassification(args, device='cuda:0', pretrained_model=pretrained_model,tokenizer=tokenizer)
    optimizer = torch.optim.Adam(model.prompt_encoder.parameters(), lr=1e-5, weight_decay=0.0005)
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)
    perf = 0
    best_dev = 0.0
    early_stop = 20
    for epoch_idx in tqdm(range(args.epoch)):
        total_train_pred = []
        total_train_labels = []
        for batch_idx, batch_data in enumerate(tqdm(train_loader)):
            model.train()
            if args.task_name == 'SST-2':
                sentences, labels = batch_data
                loss, pred = model(sentences, labels)
            elif args.task_name == 'SNLI':
                sentences_a, sentences_b, labels = batch_data
                loss, pred = model(sentences_a, labels, sentences_b)
            total_train_pred += pred
            total_train_labels += labels.tolist()
            
            if batch_idx % args.print_num == 0 and batch_idx > 0:
                acc = (torch.tensor(total_train_labels).long() == torch.tensor(total_train_pred).long()).sum() / len(total_train_labels)
                print(f'train_loss: {loss.item()}, train_acc: {acc}') 
                total_train_pred = []
                total_train_labels = []

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch_idx % args.eval_num == 0 and batch_idx > 0:
                with torch.no_grad():
                    model.eval()
                    total_dev_loss = 0
                    total_dev_pred = []
                    total_dev_labels = []

                    if args.task_name == 'SST-2':
                        for batch_idx, batch_data in tqdm(enumerate(dev_loader)):
                            sentences, labels = batch_data
                            dev_loss, dev_pred = model(sentences, labels, epoch_idx, batch_idx)
                            total_dev_loss += dev_loss.item()
                            total_dev_pred += dev_pred
                            total_dev_labels += labels.tolist()
                    elif args.task_name == 'SNLI':
                        for batch_idx, batch_data in tqdm(enumerate(dev_loader)):
                            sentences_a, sentences_b, labels = batch_data
                            dev_loss, dev_pred = model(sentences_a, labels, sentences_b, epoch_idx, batch_idx)
                            total_dev_loss += dev_loss.item()
                            total_dev_pred += dev_pred
                            total_dev_labels += labels.tolist()

                dev_acc = (torch.tensor(total_dev_labels).long() ==  torch.tensor(total_dev_pred).long()).sum() / len(total_dev_labels)
                total_dev_loss = total_dev_loss/(batch_idx+1)
                print(f'dev_loss: {total_dev_loss}, dev_acc: {dev_acc}')
                if dev_acc > best_dev:
                    best_dev = dev_acc
                    best_ckpt = {'embedding': model.prompt_encoder.state_dict(),'best_dev': best_dev}
                    if not os.path.exists('./saved_model_{}/'.format(args.task_name)):
                        os.makedirs('./saved_model_{}/'.format(args.task_name))
                    torch.save(best_ckpt, './saved_model_{}/{}_{}.pt'.format(args.task_name, epoch_idx, str(round(float(best_dev),4))))
        my_lr_scheduler.step() 

train() 