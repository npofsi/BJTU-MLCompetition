from dataloader import load_file
from transformers import AutoModelForMaskedLM,AutoTokenizer
import os
from transformers.optimization import get_scheduler

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


from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
import evaluate
import os
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
import torch
import random
import argparse
import sys
from dataloader import load_file
from dataloader import SST2Dataset, SNLIDataset
from tqdm.auto import tqdm

torch.cuda.empty_cache()

def construct_generation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default='SNLI')
    parser.add_argument("--data_path", type=str, default='.\\data')
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

train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True)
eval_dataloader = DataLoader(dev_set, batch_size=32, shuffle=False, drop_last=False)
 


# tokenized_datasets = tokenized_datasets.remove_columns(["text"])
# tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# tokenized_datasets.set_format("torch")

# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))



# train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
# eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)










device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_embedding_layer(model):
    embeddings = model.get_input_embeddings()
    return embeddings

def calculate_metrics(pred_ids, label_id):
    metrics = {}
    return metrics


class FineTuneForClassification(torch.nn.Module):
    def __init__(self, args, device, pretrained_model, tokenizer):
        super().__init__()
        self.device = device
        self.args = args
        self.tokenizer = tokenizer
        self.model = pretrained_model
        self.model = self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True
        self.embeddings = get_embedding_layer(self.model)

        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
        # self.tokenizer.add_special_tokens({'additional_special_tokens': ['[PROMPT]']})
        # self.pseudo_token_id = self.tokenizer.get_vocab()['[PROMPT]']
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id
        # self.prompt_encoder = PromptEncoderPrefixLSTM(self.hidden_size, self.tokenizer, self.device)
        # self.prompt_encoder = self.prompt_encoder.to(self.device)
        self.ce_loss = torch.nn.CrossEntropyLoss()
        if args.task_name == 'SST-2':
            self.label_map = {0:'bad',1:'great'}
        elif args.task_name == 'SNLI':
            self.label_map = {0: 'No', 1: 'Maybe', 2:'Yes'}



    def tokenize(self, query, tokens=0):
        token_ids = self.tokenizer.encode(''+query, add_special_tokens=True)
        return token_ids

    def get_query(self, sentence,tokens=0, sentence2=None):
        if self.args.task_name == 'SST-2':
            query = f'{sentence}.'# It is {self.tokenizer.mask_token} 
        elif self.args.task_name == 'SNLI':
            query = f'{sentence}, {sentence2}'#{self.tokenizer.mask_token}
        #query_Lstr=self.tokenizer.tokenize(query) 
        #print(self.tokenizer.encode(query_Lstr, add_special_tokens=True))
        #print(query)
        
        return tokenizer.encode_plus(
                query,                  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                padding='max_length',
                max_length = 45,           # Pad & truncate all sentences.
                pad_to_max_length=True,
                truncation=True,
                return_attention_mask=True,   # Construct attn. masks.
                return_tensors='pt',     # Return pytorch tensors.
            )#self.tokenize(query, tokens)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def forward(self, sentences, labels, sentences2=None,epoch_i=-1,batch_i=-1): 
        #self.model.zero_grad()  
        bz = len(sentences)
        input_ids = []
        attention_masks = []
        queries = []
        # For every sentence...
        for sent_idx, sent in enumerate(sentences):

            if sentences2 is not None:
                encoded_dict = self.get_query(sent,0,sentences2[sent_idx])
            else:
                encoded_dict = self.get_query(sent, 0)
            # 将编码后的文本加入到列表  
            input_ids.append(encoded_dict['input_ids'])
            
            # 将文本的 attention mask 也加入到 attention_masks 列表
            attention_masks.append(encoded_dict['attention_mask'])

        # 将列表转换为 tensor
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = labels.clone().detach()

        # 输出第 1 行文本的原始和编码后的信息
        # print('Original: ', sentences[0])
        # print('Token IDs:', input_ids[0])

            # if self.args.task_name == 'SST-2':
            #     queries = [torch.LongTensor(self.get_query(
            #         sentences[i],0)).squeeze(0) for i in range(bz)]
            # elif self.args.task_name == 'SNLI':
            #     queries = [torch.LongTensor(self.get_query(
            #         sentences[i],0,sentences2[i])).squeeze(0) for i in range(bz)]
        
        # print((queries))
        
        #queries = self.tokenizer.encode_plus()
        #queries = pad_sequence(
        #   queries, True, padding_value=self.pad_token_id,).long().to(self.device)
        #print("[===========]\n{}:===={}",queries.shape)
        label_ids = torch.LongTensor(labels).reshape((bz, -1)).to(self.device)
        attention_mask = queries != self.pad_token_id
        #inputs_embeds = self.embed_input(queries)
      
        #label_mask = (queries == self.tokenizer.mask_token_id).nonzero().reshape(bz, -1)[:, 1].unsqueeze(1).to(self.device)  # bz * 1
        query_output = self.model(torch.Tensor(input_ids).to(device),
                            attention_mask=attention_masks.to(self.device).bool(),
                            output_hidden_states=True,
                            return_dict=True)
        
        logits = query_output['logits']
        if self.args.task_name == 'SST-2':
            interested_logits = logits[:, :,
                                       [self.tokenizer.convert_tokens_to_ids('bad'),
                                        self.tokenizer.convert_tokens_to_ids('great')]]
        elif self.args.task_name == 'SNLI':
            interested_logits = logits[:, :,
                                       [self.tokenizer.convert_tokens_to_ids('Yes'),
                                        self.tokenizer.convert_tokens_to_ids('Maybe'),
                                        self.tokenizer.convert_tokens_to_ids('No')]]
        
        pred_ids = torch.argsort(interested_logits, dim=2, descending=True)
        batch_interested_logits = []
        predicted_label = [] 
        
        
        for i in range(bz):
            pred_seq = pred_ids[i, 0].tolist()
            predicted_label.append(pred_seq[0])
            batch_interested_logits.append(
                interested_logits[i, 0].cpu())

        
        batch_interested_logits = torch.stack(
            batch_interested_logits).to(self.device)
        #print(batch_interested_logits.shape)
        if epoch_i!=-1 and batch_i!=-1 and epoch_i == batch_i:
            print("Save log")
            lines=[]
            
            for i,d in enumerate(sentences):
                indices = torch.LongTensor([i, 1, 1]).cpu()
                batch_interested_logits_c=batch_interested_logits.clone().detach().cpu()
                t=torch.index_select(batch_interested_logits_c, 0, indices)
                if sentences2 is not None:
                    lines.append(""+str(predicted_label[i])+"|"+str(labels[i])+", "+str(sentences[i])+" + "+str(sentences2[i])+", "+str(t)+"\n")
                else:
                    lines.append(""+str(predicted_label[i])+"|"+str(labels[i])+", "+str(sentences[i])+", "+str(t)+"\n")


            fo = open("log\\"+args.task_name+"_batchlog_"+str(epoch_i)+"_"+str(batch_i)+".txt", mode="w", encoding="utf-8")
            fo.writelines(lines)

        loss = self.ce_loss(batch_interested_logits, label_ids.squeeze(1))
        return loss, predicted_label


def train():
    num_training_steps = args.epoch * len(train_dataloader)
    model = FineTuneForClassification(args,device,pretrained_model,tokenizer)
    optimizer = AdamW(model.model.parameters(), lr=5e-5)
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=0.98)
    #print(len(train_dataloader))
    perf = 0
    best_dev = 0.0
    early_stop = 20
    
    progress_bar = tqdm(range(num_training_steps))
    
    model.to(device)
    
    for epoch_idx in tqdm(range(args.epoch)):
        total_train_pred = []
        total_train_labels = []
        for batch_idx, batch_data in enumerate(tqdm(train_dataloader)):
            model.train()
            if args.task_name == 'SST-2':
                sentences, labels = batch_data
                loss, pred = model(sentences, labels)
            elif args.task_name == 'SNLI':
                sentences_a, sentences_b, labels = batch_data
                loss, pred = model(sentences_a, labels, sentences_b)
            #loss.requires_grad = True
            total_train_pred += pred
            total_train_labels += labels.tolist()
            
            if batch_idx % args.print_num == 0 and batch_idx > 0:
                acc = (torch.tensor(total_train_labels).long() == torch.tensor(
                    total_train_pred).long()).sum() / len(total_train_labels)
                print(f'train_loss: {loss.item()}, train_acc: {acc}')
                total_train_pred = []
                total_train_labels = []

            # with torch.no_grad():
                
                # total_dev_pred = []
                # total_dev_labels = []
            #torch.set_grad_enabled(True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
            optimizer.step()
            # my_lr_scheduler.step()
            optimizer.zero_grad()



            if batch_idx % args.eval_num == 0 and batch_idx > 0:
                with torch.no_grad():
                    model.eval()
                    total_dev_loss = 0
                    total_dev_pred = []
                    total_dev_labels = []

                    if args.task_name == 'SST-2':
                        for batch_idx, batch_data in tqdm(enumerate(eval_dataloader)):
                            sentences, labels = batch_data
                            dev_loss, dev_pred = model(sentences, labels, epoch_i=epoch_idx, batch_i=batch_idx)
                            total_dev_loss += dev_loss.item()
                            total_dev_pred += dev_pred
                            total_dev_labels += labels.tolist()
                    elif args.task_name == 'SNLI':
                        for batch_idx, batch_data in tqdm(enumerate(eval_dataloader)):
                            sentences_a, sentences_b, labels = batch_data
                            dev_loss, dev_pred = model(
                                sentences_a, labels, sentences_b, epoch_i=epoch_idx, batch_i=batch_idx)
                            total_dev_loss += dev_loss.item()
                            total_dev_pred += dev_pred
                            total_dev_labels += labels.tolist()

                dev_acc = (torch.tensor(total_dev_labels).long() == torch.tensor(
                    total_dev_pred).long()).sum() / len(total_dev_labels)
                total_dev_loss = total_dev_loss/(batch_idx+1)
                print(f'dev_loss: {total_dev_loss}, dev_acc: {dev_acc}')
                if dev_acc > best_dev:
                    best_dev = dev_acc
                    best_ckpt = {
                        'embedding': 0, 'best_dev': best_dev}
                    if not os.path.exists('./saved_modelf_{}/'.format(args.task_name)):
                        os.makedirs('./saved_modelf_{}/'.format(args.task_name))
                    torch.save(best_ckpt, './saved_modelf_{}/{}_{}.pt'.format(
                        args.task_name, epoch_idx, str(round(float(best_dev), 4))))
        my_lr_scheduler.step()

train()