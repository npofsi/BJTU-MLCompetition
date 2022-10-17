import torch
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support,f1_score
import math
from datasets import load_dataset
from transformers import BertTokenizer,BertModel,AdamW

import warnings
warnings.filterwarnings("ignore")
BASE = os.getcwd() + "/bert-base"#文件位置目录





#定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.label2id = {"科技":0,"股票":1,"教育":2,"财经":3,"娱乐":4} 
        #存放训练集 验证集和测试集的路径
        data_files={
            "train": [f"{BASE}/data/extract/{label}-train.csv" for label in  self.label2id],
            "dev": [f"{BASE}/data/extract/{label}-dev.csv" for label in  self.label2id],
            "test": [f"{BASE}/data/extract/{label}-test.csv" for label in  self.label2id],
        } 
        #读取数据，delimiter是每行的分隔符，column_names是文件数据的列名
        self.dataset = load_dataset('csv', data_files=data_files, delimiter='\t', column_names=[ "label","title", "content"], split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i): #迭代生成每条数据
        text = self.dataset[i]['title'] #一个i就是一条数据，将标题作为训练的文本，也可以增加content
        label = self.label2id[self.dataset[i]['label']]  #把文字的label转换成id

        return text, label

#数据加载函数
def collate_fn(data): 
    # print(data)
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    #编码，把中文变成id
    data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                    #当句子长度大于max_length时,截断
                                   truncation=True,
                                   padding='max_length', #一律补零到max_length长度
                                   max_length=50,
                                   return_tensors='pt', #可取值tf,pt,np,默认为返回list
                                   return_length=True)  #返回length 标识长度

    #input_ids:编码之后的数字
    input_ids = data['input_ids']
    #attention_mask:是补零的位置是0,其他位置是1
    attention_mask = data['attention_mask']
    #token_type_ids 第一个句子和特殊符号的位置是0,第二个句子的位置是1
    token_type_ids = data['token_type_ids']

    #标签
    labels = torch.LongTensor(labels)

    #print(data['length'], data['length'].max())
    return input_ids, attention_mask, token_type_ids, labels

#定义基于bert的文本分类模型
class MyBertClassification(torch.nn.Module):
    def __init__(self,num_label,bertmodel,is_finetune):
        super(MyBertClassification,self).__init__()
        # 定义分类器
        self.fc = torch.nn.Linear(768, num_label)
        # 定义bert
        self.bert = bertmodel
        # bert是否参与微调
        self.is_finetune=is_finetune

    #指标计算函数
    def compute(self,labels, preds):
        precision, recall, macro_f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        micro_f1 = precision_recall_fscore_support(labels, preds, average='micro')[2]
        # if precision + recall < 0.01:
        #     precision = 0.01
        # f1 = 2 * precision * recall / (precision + recall)
        if math.isnan(macro_f1):
            macro_f1 = 0
        if math.isnan(micro_f1):
            micro_f1 = 0

        return {
            "micro_f1":micro_f1,
            'macro_f1': macro_f1,
            'precision': precision,
            'recall': recall
        }
    
    #前向传播过程
    def forward(self, input_ids, attention_mask, token_type_ids):
        if not self.is_finetune:
            with torch.no_grad():
                #得到bert最后一层的每句话的输出向量，shape=(bs,maxlen,768)
                out = self.bert(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        else:
            out = self.bert(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)

        out = self.fc(out.last_hidden_state[:, 0]) #取出[CLS]的首部信息作为分类器的输入

        out = out.softmax(dim=1) #softmax转换成概率

        return out

def my_train(model,optimizer,criterion,train_loader,dev_loader,num_epoch,save_path,early_stop,device,dev_batch):
    early_stop_flag = 0 

    #记录最好结果的模型epoch
    best_epoch = 0 
    best_batch = 0
    #最好的模型在验证集上的指标
    best_val_f_macro = 0
    
    for epoch in range(num_epoch):

        for batch_id, (input_ids, attention_mask, token_type_ids,
                labels) in enumerate(train_loader):
            #标识模型训练
            model.train()

            input_ids=input_ids.to(device)
            attention_mask=attention_mask.to(device)
            token_type_ids=token_type_ids.to(device)
            labels=labels.to(device)

            # print(input_ids.is_cuda)
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)  #(batch_size,outdim)
            #计算损失
            train_loss = criterion(out, labels)
            #反向传播
            train_loss.backward()
            #参数更新
            optimizer.step()
            #梯度清零
            optimizer.zero_grad()

            #每40batch计算验证集指标
            if (batch_id+1) % dev_batch == 0:
                #模型验证
                model.eval()
                #记录全部dev集的预测类别和标签类别
                Labels=torch.tensor([]);Pres=torch.tensor([])
                for i, (input_ids, attention_mask, token_type_ids,
                labels) in enumerate(dev_loader):
                    input_ids=input_ids.to(device);attention_mask=attention_mask.to(device);token_type_ids=token_type_ids.to(device)
                    out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
                    loss = criterion(out.cpu(), labels)
                    out = out.argmax(dim=1)
                    out.detach_() 

                    #拼接
                    Labels=torch.concat([Labels,labels])
                    Pres=torch.concat([Pres,out.cpu()])
                #指标计算
                zb = model.compute(Labels, Pres)
                # print(i,loss.item(), zb)
                      

                #记录最优结果和保存模型
                if zb['macro_f1'] > best_val_f_macro:
                    best_val_f_macro = zb['macro_f1']
                    best_epoch = epoch
                    best_batch = batch_id
                    early_stop_flag = 0
                    # 保存本轮训练结果
                    torch.save({'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':i, 'batch':batch_id}, save_path) 
                    print(f"*** epoch{epoch + 1},batch{batch_id+1} train loss {train_loss.item()}, dev loss {loss.item()}, dev micro_f1 {zb['micro_f1']}, dev macro_f1 {zb['macro_f1']}, best f1-macro now:{best_val_f_macro}")
                else:
                    early_stop_flag += 1
                    print(f"*** epoch{epoch + 1},batch{batch_id+1} train loss {train_loss.item()}, dev loss {loss.item()}, dev micro_f1 {zb['micro_f1']}, dev macro_f1 {zb['macro_f1']}, best f1-macro now:{best_val_f_macro}")
                    #早停
                    if early_stop_flag == early_stop:
                        print(f'\nThe model has not been improved for {early_stop} rounds. Stop early!')
                        return model,best_epoch,best_batch
    return model,best_epoch,best_batch,zb

    

#测试
def my_test(save_path,num_label,bertmodel,is_finetune,device):
    #加载最优模型       
    test_model = MyBertClassification(num_label,bertmodel,is_finetune).to(device)
    test_model.load_state_dict(torch.load(save_path)['net'])

    #测试集加载
    test_dataset = Dataset('test')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                     batch_size=32,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)
    model.eval()
    test_out=torch.tensor([]);test_label = torch.tensor([])
    for batchid, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(test_loader):
        input_ids=input_ids.to(device)
        attention_mask=attention_mask.to(device)
        token_type_ids=token_type_ids.to(device)

        #测试过程没有梯度
        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids) #(bs,outdim)

        out = out.argmax(dim=1)
        test_label=torch.concat([test_label,labels])
        test_out=torch.concat([test_out,out.cpu()])

    zb = model.compute(test_label, test_out)
    print(f"*** test ***")     
    print(zb) 
    return  test_out  



#--------------------- 参数设置 模型初始化 --------------------------------------------------
# GPU or CPU
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
# 学习率,不参与fine-tune时 可以设置5e-5
lr=1e-6 
# bert是否参与下游的微调
is_finetune=True
# 早停，如果10次内指标不再上升就停止训练
early_stop=10
# 模型保存路径
save_path=f"{BASE}/result/model.pt"
# 训练epoch
num_epoch=5
# 标签类别
num_label=5
# batch大小 可以适当增大或减小
batch_size=32
# 每dev_batch个batch后进行验证集的指标评估
dev_batch=40 

#加载预训练模型Bert
pretrained = BertModel.from_pretrained(f'{BASE}/model/bert-base-chinese').to(device) # huggingface模型
#定义自己的文本分类模型
model = MyBertClassification(num_label,pretrained,is_finetune).to(device)
#加载字典和分词工具
tokenizer = BertTokenizer.from_pretrained(f'{BASE}/model/bert-base-chinese')

#数据加载器
train_dataset = Dataset('train')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                     batch_size=batch_size,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)
dev_dataset = Dataset('dev')
dev_loader = torch.utils.data.DataLoader(dataset=dev_dataset,
                                     batch_size=batch_size,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)

#优化器定义
optimizer = AdamW(model.parameters(), lr=lr)  
#损失函数
criterion = torch.nn.CrossEntropyLoss().to(device)

#训练
my_train(model,optimizer,criterion,train_loader,dev_loader,num_epoch,save_path,early_stop,device,dev_batch)

#测试
my_test(save_path,num_label,pretrained,is_finetune,device)
#--------------------------------------------------------------------------------------------






