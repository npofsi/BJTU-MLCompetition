from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import sys
sys.path.append("/data/qingyang/Baseline/BiLSTM-Baseline/")
from BiLSTM_Baseline import load_title_label, fenci, train_path

cut_save_path = "data/train_cut.txt"

title_cut_list = fenci(load_title_label(train_path))

with open(cut_save_path, 'w', encoding='utf-8') as fw:
    for new_title in title_cut_list:
        fw.write(' '.join(new_title[0]))
        fw.write('\n')
    fw.write('<PAD>\n')
    fw.write('<UNK>\n')

#inp:分好词的文本
#outp1:训练好的模型
#outp2:得到的词向量

inp=cut_save_path
outp1='BiLSTM-Baseline/word2vector/model/w2v.model'
outp2='BiLSTM-Baseline/word2vector/model/w2v.vector'

'''
LineSentence(inp)：格式简单：一句话=一行; 单词已经过预处理并被空格分隔。
vector_size：是每个词的向量维度； 
window：是词向量训练时的上下文扫描窗口大小，窗口为5就是考虑前5个词和后5个词； 
min-count：设置最低频率，默认是5，如果一个词语在文档中出现的次数小于5，那么就会丢弃； 
workers：是训练的进程数（需要更精准的解释，请指正），默认是当前运行机器的处理器核数。这些参数先记住就可以了。
sg ({0, 1}, optional) – 模型的训练算法: 1: skip-gram; 0: CBOW
alpha (float, optional) – 初始学习率
iter (int, optional) – 迭代次数，默认为5
'''
model = Word2Vec(LineSentence(inp), vector_size=200, window=5, min_count=1, workers=20, sg=1)
model.save(outp1)
model.wv.save_word2vec_format(outp2, binary=False)



