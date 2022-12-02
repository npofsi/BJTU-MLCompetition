import os
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys
import torch
import json

def load_file(path, task_name, data_type):
    file = open(os.path.join(path, task_name, f'{data_type}.tsv'), 'r', encoding='utf-8')
    if task_name == 'SST-2':
        texts = []
        labels = []
        lines = file.readlines()
        lines = lines[1:]
        for line in lines:
            text, label = line.replace('\n','').split('\t')
            texts.append(text)
            labels.append(label)
        return texts, labels
    elif task_name == 'SNLI':
        text_as = []
        text_bs = []
        labels = []
        lines = file.readlines()
        lines = lines[1:]
        for line in lines:
            text_a,text_b,label = line.replace('\n','').split('\t')
            text_as.append(text_a)
            text_bs.append(text_b)
            labels.append(label)
        return text_as, text_bs, labels
    


class SST2Dataset(Dataset):
    def __init__(self,  args, texts, labels):
        super().__init__()
        self.args = args
        self.texts = texts
        self.labels = [int(lb) for lb in labels]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


class SNLIDataset(Dataset):
    def __init__(self, args, sentence_1, sentence_2, label):
        super().__init__()
        self.args = args
        # self.label_map = {
        #     'neutral':'Maybe',
        #     'contradiction':'No',
        #     'entailment':'Yes'
        # }
        self.label_map = {
            'entailment':0,
            'neutral':1,
            'contradiction':2,
        }
        self.sentence_1 = []
        self.sentence_2 = []
        self.label = []
        for idx in range(len(label)):
            if label[idx] in self.label_map.keys():
                self.sentence_1.append(sentence_1[idx])
                self.sentence_2.append(sentence_2[idx])
                self.label.append(label[idx])
 
        self.label = [int(self.label_map[lb]) for lb in self.label] 
        

    def __len__(self):
        return len(self.sentence_1)

    def __getitem__(self, i): 
        return self.sentence_1[i], self.sentence_2[i], self.label[i]
 