from gensim.models import Word2Vec

model_path = 'BiLSTM-Baseline/word2vector/model/w2v.model'

model = Word2Vec.load(model_path)

print(model.wv['<UNK>'])

