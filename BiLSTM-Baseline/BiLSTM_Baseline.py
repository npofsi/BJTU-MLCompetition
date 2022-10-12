from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
import jieba
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from sklearn import metrics 

import warnings
warnings.filterwarnings("ignore")

# 一些超参数
# 每批次训练样本数量
batch_size = 128
# 每个title最多有多少个词送入模型
max_len = 10
# 对词语进行编码的长度
# ⬇ 意思每个词语由200个 数值 来表示
embedding_dim=200
# BiLSTM隐层层数
hidden_size=64
# 学习率
lr = 0.01
# 训练轮次
epoch_num = 5
# 训练设备（有GPU就使用GPU否则使用CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

# 标签对应的id（计算过程中使用id进行计算
label2idx = {
    '科技' : 0, 
    '财经' : 1, 
    '教育' : 2, 
    '股票' : 3, 
    '娱乐' : 4, 
}

# 数据集的路径
train_path = '/data/yanxu/thucnews/train_set.csv'
test_path = '/data/yanxu/thucnews/test_set.csv'


# load stopwords set
# 载入停用词，停用词对模型训练帮助很小但会增加训练时间，所以进行一个删除
stopword_set = set()
with open(r"BiLSTM-Baseline/jieba/stopwords_simple.txt", 'r', encoding='utf-8') as stopwords:
    for stopword in stopwords:
        stopword_set.add(stopword.strip('\n'))


class MyDatasets(Dataset):
    """
    构建数据集。
    继承Dataset的类要实现以下三个函数：
    1. __init__ 初始化函数
    2. __getitem__ 使类的实例可以按照下标访问 -> mydatasets[0]
    3. __len__ 使类的实例可以被len()函数衡量 -> len(mydatasets)
    """
    def __init__(self, data, word2idx, max_len) -> None:
        self.data = data

        for index in range(len(data)):
            # 将每个词语都根据词典转换成对应的id
            # 不存在词典中的词语会用<UNK>替代
            idx = [ word2idx.get(word, 0) for word in data[index][0]]
            if len(idx) >= max_len:
                # title所含词语大于最大长度，截断
                self.data[index][0] = idx[:max_len]
            else:
                # title所含词语小于最大长度，用<PAD>补全
                self.data[index][0] = idx + [1] * (max_len - len(idx))
    
    def __getitem__(self, index):
        # 转换训练数据和label的格式为torch.tensor
        x = torch.tensor(self.data[index][0], dtype=torch.long)
        label = torch.tensor(self.data[index][1], dtype=torch.long)
        return x, label

    def __len__(self):
        return len(self.data)


class MyBiLSTM(nn.Module):
    """
    模型类
    1.__init__ 初始化函数，定义类的结构
    2.__forward__ 前向传播函数
    """
    def __init__(self, num_embeddings, embedding_dim, hidden_size, output_size) -> None:
        """
        num_embeddings: 词典中词语的数量（一共需要编码多少个词
        embedding_dim: 每个词用多少个数值进行编码
        hidden_size: 隐层数量
        output_size: 输出维度（分类问题一般为分类的种类个数
        """
        super().__init__()
        # 编码层
        # 输入 (batch_size, seq_len)
        # 输出 (batch_size, seq_len, embedding_dim)
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        # 输入 (batch_size, seq_len, embedding_dim)
        # 输出 (batch_size, sql_len, 2*hidden_size)
        self.bilstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        # 输入 (batch_size, sql_len, 2*hidden_size)
        # 输出 (batch_size, sql_len, output_size)
        self.decoder = nn.Linear(2*hidden_size, output_size)

    
    def forward(self, input):
        # 编码
        embedding = self.embedding(input)
        # BiLSTM，模型的主要部分
        out, (h, c) = self.bilstm(embedding)
        # 解码，将BiLSTM的输入转换为输出结果
        out = self.decoder(out)
        return out


def load_title_label(path: str) -> list:
    """
    读取数据文件，并将label处理为对应的序列号，返回一个二维列表
    """
    examples = pd.read_table(path, sep='\t', names=['label', 'title', 'text']) # 读取data文件
    data = []
    for item in zip(examples['title'], examples['label']): # 打包title列 和 label列
        item = list(item)
        item[1] = label2idx[item[1]] # 将label处理为对应的 index
        data.append(item)
    return data


def fenci(data: list) -> list:
    """
    对标题进行分词操作
    """
    data_cut = []
    # 遍历每一个数据
    for example in data:        
        # 对title进行分词
        words = set(jieba.cut(example[0]))
        # 去除停用词
        temp = []
        for item in words:
            if item not in stopword_set:
                temp.append(item)
        words = temp
        # 保存分词后的结果
        data_cut.append([words, example[1]])
    
    return data_cut


def build_dictionary(data: list) -> dict and dict and int:
    """
    构造字典，使每一个词语都有对应的编号
    """
    word2idx = {'<UNK>': 0, '<PAD>': 1}
    idx2word = {0: '<UNK>', 1: '<PAD>'}
    word_set = set()
    # 读取传入的数据，读入每一个词并去重
    for example in data:
        for word in example[0]:
            if len(word):
                word_set.add(word)
    
    # 构建字典
    for word in word_set:
        word2idx[word] = len(word2idx)
        idx2word[len(idx2word)] = word
    
    return word2idx, idx2word, len(word2idx)


def train(train_loader: DataLoader, model: MyBiLSTM):
    # 绑定运行设备
    model.to(device)
    # 切换成训练模式
    model.train()
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 所有训练数据训练epoch_num轮
    for epoch in range(epoch_num):
        # 记录每一轮的平均loss
        loss_each_epoch = 0
        for index, (input, label) in tqdm(enumerate(train_loader)):
            # 运算数据绑定设备
            input = input.to(device)
            label = label.to(device)
            # 前向传播
            out = model(input)
            # （batch_size, sql_len, output_size）
            # 取sql的第一个token(词)作为这句话的预测结果
            out = out[:,0,:].squeeze(-1)
            # 计算损失
            loss = criterion(out, label)
            
            loss_each_epoch += loss.item()
            # 反向传播
            loss.backward()
            
            # 梯度下降，更新参数
            optimizer.step()
            # 梯度清除， 为下一次梯度下降做准备
            optimizer.zero_grad()

        print('epoch: ', epoch, ' loss:', loss_each_epoch / (index+1))

    print('训练完毕')



def evaluate(data_loader: DataLoader, model: MyBiLSTM):
    # 绑定设备
    model.to(device)
    # 设定模型状态
    model.eval()

    acc = precision = recall = f1 = 0
    
    # 对所有的测试数据进行一次测试
    for index, (input, label) in tqdm(enumerate(data_loader)):
        # 同训练部分
        input = input.to(device)
        label = label.to(device)
        
        out = model(input)
        out = out[:,0,:].squeeze(-1)
        
        # 使用softmax激活函数，使各类概率和为1
        # 使用argmax返回值最大的元素的下标
        # dim = 1 指第一个纬度，也就是行
        prediction = torch.argmax(F.softmax(out, dim=1), dim=1)
        
        # 计算指标要把 数值变成 cpu 上，在进行计算
        # label.detach().cpu().numpy() -> 就是把GPU上的数据，转移到CPU上
        acc += metrics.accuracy_score(label.detach().cpu().numpy(), prediction.detach().cpu().numpy())
        precision += metrics.precision_score(label.detach().cpu().numpy(), prediction.detach().cpu().numpy(), average='macro')
        recall += metrics.recall_score(label.detach().cpu().numpy(), prediction.detach().cpu().numpy(), average='macro')
        f1 += metrics.f1_score(label.detach().cpu().numpy(), prediction.detach().cpu().numpy(), average='macro')
    
    # 求几次batch的均值
    acc /= index + 1
    precision /= index + 1
    recall /= index + 1
    f1 /= index + 1
    print('acc', acc, 'precision:', precision, ' recall:', recall, ' f1:', f1)

    model.train()


if __name__ == '__main__':
    # 获取数据集
    train_data = load_title_label(train_path)
    test_data = load_title_label(test_path)
    # 对数据集的title进行分词
    train_data_cut = fenci(train_data)
    test_data_cut = fenci(test_data)
    # 利用分词结果，构建词典
    word2idx, idx2word, num_embeddings = build_dictionary(train_data_cut)
    # 根据词典，构造MyDatasets
    train_iter = MyDatasets(data=train_data_cut, word2idx=word2idx, max_len=max_len)
    test_iter = MyDatasets(data=test_data_cut, word2idx=word2idx, max_len=max_len)
    # 构建数据集载入类
    train_loader = DataLoader(dataset=train_iter, batch_size=batch_size ,shuffle=True)
    test_loader = DataLoader(dataset=test_iter, batch_size=batch_size ,shuffle=True)
    # 定义模型
    model = MyBiLSTM(
        num_embeddings=num_embeddings, 
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        output_size=len(label2idx)
    )
    # 训练
    train(train_loader=train_loader, model=model)
    # 测试
    evaluate(data_loader=test_loader, model=model)