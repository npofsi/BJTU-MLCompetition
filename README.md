# Baseline
## 数据集地址
链接：https://pan.baidu.com/s/1ZBbkcOJLbJUse37CAWnCQg 
提取码：0pgu
## 安装环境

## Bert-Baseline
- 文本分类-bert.ipynb 仅供大家参考学习
- bert_base.py 真正的baseline文件
## BiLSTM-Baseline
- jieba/ 存放jieba分词的停用词文件
- BiLSTM_Baseline.py baseline文件
- 运行时要记得修改数据路径和停用词路径
- 使用预训练模型进行embedding层初始化时
    - 要先运行BiLSTM-Baseline/word2vector/train.py
