# Baseline


## 安装环境
1. 安装anaconda
- anaconda安装：https://www.cnblogs.com/sui776265233/p/11453004.html
2. 安装需要的python库
- 虚拟环境管理教程：https://blog.csdn.net/weixin_43267006/article/details/124847368
- 创建并切换到虚拟环境后，安装python库
```cmd
conda create -n your_vritual_env_name python=3.8
conda activete your_vritual_env_name
pip install -r requirements.txt

``` 
3. 如果使用GPU训练，还需安装cuda
- https://blog.csdn.net/m0_45447650/article/details/123704930
## Bert-Baseline
- 文本分类-bert.ipynb 仅供大家参考学习
- bert_base.py 真正的baseline文件
## BiLSTM-Baseline
- jieba/ 存放jieba分词的停用词文件
- BiLSTM_Baseline.py baseline文件
- 运行时要记得修改数据路径和停用词路径
- 使用预训练模型进行embedding层初始化时
    - 要先运行BiLSTM-Baseline/word2vector/train.py
