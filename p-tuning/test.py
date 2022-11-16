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
    parser.add_argument("--ckpt", type=str, default='16_0.5479.pt')

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
    test_texts, test_labels = load_file(data_path,task_name,'test')
    test_set = SST2Dataset(args, test_texts, test_labels)

elif args.task_name == 'SNLI':
    # 由于没有带标签的测试集，可用dev做验证实验
    test_texts_a, test_texts_b, test_labels = load_file(data_path,task_name,'dev')
    test_set = SNLIDataset(args, test_texts_a, test_texts_b, test_labels)

test_loader = DataLoader(test_set, batch_size=32, shuffle=False, drop_last=False)
 

def test():
    model = PTuneForClassification(args, device='cuda:0', pretrained_model=pretrained_model,tokenizer=tokenizer)
    checkpoint = torch.load(f'./saved_model_{args.task_name}/{args.ckpt}') 
    model.prompt_encoder.load_state_dict(checkpoint['embedding'])
   
    with torch.no_grad():
        model.eval()
        total_test_loss = 0
        total_test_pred = []
        total_test_labels = []

        if args.task_name == 'SST-2':
            for batch_idx, batch_data in tqdm(enumerate(test_loader)):
                sentences, labels = batch_data
                test_loss, test_pred = model(sentences, labels)
                total_test_loss += test_loss.item()
                total_test_pred += test_pred
                total_test_labels += labels.tolist()
        elif args.task_name == 'SNLI':
            for batch_idx, batch_data in tqdm(enumerate(test_loader)):
                sentences_a, sentences_b, labels = batch_data
                test_loss, test_pred = model(sentences_a, labels, sentences_b)
                total_test_loss += test_loss.item()
                total_test_pred += test_pred
                total_test_labels += labels.tolist()

                test_acc = (torch.tensor(total_test_labels).long() ==  torch.tensor(total_test_pred).long()).sum() / len(total_test_pred)
                total_test_loss = total_test_loss/(batch_idx+1)
        print(f'test_loss: {total_test_loss}, test_acc: {test_acc}')
               
test() 