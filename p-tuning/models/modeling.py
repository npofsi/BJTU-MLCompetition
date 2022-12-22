import torch
from torch.nn.utils.rnn import pad_sequence
from os.path import join
from transformers import AutoTokenizer
import sys
from models.prompt_encoder import PromptEncoderPrefixLSTM
from tqdm import tqdm
import numpy as np
import torch.nn as nn

def get_embedding_layer(model):
    embeddings = model.get_input_embeddings()
    return embeddings

def calculate_metrics(pred_ids, label_id):
    metrics = {}
    return metrics

class PTuneForClassification(torch.nn.Module):
    def __init__(self, args, device, pretrained_model, tokenizer):
        super().__init__()
        self.device = device
        self.args = args
        self.tokenizer = tokenizer
        self.model = pretrained_model
        self.model = self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.embeddings = get_embedding_layer(self.model)

        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[PROMPT]']})
        self.pseudo_token_id = self.tokenizer.get_vocab()['[PROMPT]']
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id
        self.prompt_encoder = PromptEncoderPrefixLSTM(self.hidden_size, self.tokenizer, self.device)
        self.prompt_encoder = self.prompt_encoder.to(self.device)
        self.ce_loss = torch.nn.CrossEntropyLoss()
        if args.task_name == 'SST-2':
            self.label_map = {0:'bad',1:'great'}
        elif args.task_name == 'SNLI':
            self.label_map = {0: 'No', 1: 'Maybe', 2:'Yes'}

    def embed_input(self, queries):
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
        raw_embeds = self.embeddings(queries_for_embedding)
        return self.prompt_encoder(tokens=queries_for_embedding, embeddings=raw_embeds)

    def get_query(self, sentence, prompt_tokens, sentence2=None):
        if self.args.task_name == 'SST-2':
            query = f'{sentence} It is {self.tokenizer.mask_token} .'
        elif self.args.task_name == 'SNLI':
            query = f'{sentence} is {self.tokenizer.mask_token} with {sentence2}'
        return self.prompt_encoder.tokenize(query, prompt_tokens)


    def forward(self, sentences, labels, sentences2=None,epoch_i=-1,batch_i=-1): 
        bz = len(sentences)
        prompt_tokens = [self.pseudo_token_id]
        if self.args.task_name == 'SST-2':
            queries = [torch.LongTensor(self.get_query(sentences[i], prompt_tokens)).squeeze(0) for i in range(bz)]
        elif self.args.task_name == 'SNLI':
            queries = [torch.LongTensor(self.get_query(sentences[i], prompt_tokens, sentences2[i])).squeeze(0) for i in range(bz)]
        
        queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)
        label_ids = torch.LongTensor(labels).reshape((bz, -1)).to(self.device)
        attention_mask = queries != self.pad_token_id
        inputs_embeds = self.embed_input(queries)
        label_mask = (queries == self.tokenizer.mask_token_id).nonzero().reshape(bz, -1)[:, 1].unsqueeze(1).to(self.device)  # bz * 1
        query_output = self.model(inputs_embeds=inputs_embeds.to(self.device),
                            attention_mask=attention_mask.to(self.device).bool(),
                            output_hidden_states=True,
                            return_dict=True)
        logits = query_output['logits']
        if self.args.task_name == 'SST-2':
            interested_logits = logits[:,:,
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
            pred_seq = pred_ids[i, label_mask[i, 0]].tolist()
            predicted_label.append(pred_seq[0])
            batch_interested_logits.append(interested_logits[i,label_mask[i, 0]].cpu())
        batch_interested_logits = torch.stack(batch_interested_logits).to(self.device)
        #print(batch_interested_logits,label_ids
        # 
        # 
        # 
        # )

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


            fo = open("logp\\"+self.args.task_name+"_batchlog_"+str(epoch_i)+"_"+str(batch_i)+".txt", mode="w", encoding="utf-8")
            fo.writelines(lines)

        loss = self.ce_loss(batch_interested_logits, label_ids.squeeze(1))
        return loss, predicted_label 

