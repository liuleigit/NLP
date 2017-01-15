#*--- coding:utf-8---*
'''
用gensim函数库训练Word2Vec模型有很多配置参数。这里对gensim文档的Word2Vec函数的参数说明进行翻译，以便不时之需。
class gensim.models.word2vec.Word2Vec(sentences=None,size=100,alpha=0.025,window=5, min_count=5, max_vocab_size=None, sample=0.001,seed=1, workers=3,min_alpha=0.0001, sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=<built-in function hash>,iter=5,null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000)
参数：
·  sentences：可以是一个·ist，对于大语料集，建议使用BrownCorpus,Text8Corpus或·ineSentence构建。
·  sg： 用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法。
·  size：是指特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
·  window：表示当前词与预测词在一个句子中的最大距离是多少
·  alpha: 是学习速率
·  seed：用于随机数发生器。与初始化词向量有关。
·  min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
·  max_vocab_size: 设置词向量构建期间的RAM限制。如果所有独立单词个数超过这个，则就消除掉其中最不频繁的一个。每一千万个单词需要大约1GB的RAM。设置成None则没有限制。
·  sample: 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)
·  workers参数控制训练的并行数。
·  hs: 如果为1则会采用hierarchica·softmax技巧。如果设置为0（defau·t），则negative sampling会被使用。
·  negative: 如果>0,则会采用negativesamp·ing，用于设置多少个noise words
·  cbow_mean: 如果为0，则采用上下文词向量的和，如果为1（defau·t）则采用均值。只有使用CBOW的时候才起作用。
·  hashfxn： hash函数来初始化权重。默认使用python的hash函数
·  iter： 迭代次数，默认为5
·  trim_rule： 用于设置词汇表的整理规则，指定那些单词要留下，哪些要被删除。可以设置为None（min_count会被使用）或者一个接受()并返回RU·E_DISCARD,uti·s.RU·E_KEEP或者uti·s.RU·E_DEFAU·T的函数。
·  sorted_vocab： 如果为1（defau·t），则在分配word index 的时候会先对单词基于频率降序排序。
·  batch_words：每一批的传递给线程的单词的数量，默认为10000
'''


import os
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import nltk.data
from gensim.models.word2vec import Word2Vec

def load_dataset(name, nrows=None):
    dataset = {
        'unlabeled_train': 'unlabeledTrainData.tsv',
        'labeled_train': 'labeledTrainData.tsv',
        'test': 'testData.tsv'
    }
    if name not in dataset:
        raise ValueError(name)
    data_file = os.path.join('.', 'data', dataset[name])
    df = pd.read_csv(data_file, sep='\t', escapechar='\\', nrows=nrows)
    print('Number of reviews: {}'.format(len(df)))
    return df

df = load_dataset('unlabeled_train')
print df.head()

eng_stopwords = {}.fromkeys([line.rstrip() for line in open('./stopwords.txt')])
def clean_text(text, remove_stopwords=False):
    text = BeautifulSoup(text, 'html.parser').get_text() #可以去除文本中的html标签
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    if remove_stopwords:
        words = [w for w in words if w not in eng_stopwords]
    return words

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') #用来将一段话分割成句子
def print_call_counts(f):
    n = {}
    n[0] = 0
    def wrapped(*args, **kwargs):
        n[0] += 1
        if n[0] % 1000 == 1:
            print('method {} called {} times'.format(f.__name__, n[0]))
        return f(*args, **kwargs)
    return wrapped
n = 0
@print_call_counts
def split_sentences(review):
    global n
    if n == 0:
        print 'review =' + review
        n += 1
    raw_sentences = tokenizer.tokenize(review.strip().decode('utf-8'))
    sentences = [clean_text(s) for s in raw_sentences if s]
    return sentences

#df.review.apply(split_sentences) 返回类型是pandas.Series, 每条评论是序列的一维,结构如下:
#  0    [[watching, time, chasers, it, obvious, that, ..], [maybe, they, ...]
#  1    [[i, saw, this, film, about, years, ago, and, ... ]]
#其中每个[]内是一个句子
#使用sum将序列成员加起来,即所有句子放到一个list中。
# [[u'watching', u'time', u'chasers',..], [...], [i, saw, this..] ... ]
#另外,sum(sequence, start=None)中sequence是序列,是有index的,不能使用list
sentences = sum(df.review.apply(split_sentences), [])

import logging
logging.basicConfig(format='%(asctime)s:$(levelname)s:%(message)s', level=logging.INFO)

num_features = 300
min_word_count = 40
num_workers = 4
window = 10
sample = 1e-3
model_name = '{}features_{}minwords_{}window.model'.format(num_features, min_word_count, window)
print ('Training model...')
model = Word2Vec(sentences, workers=num_workers,
                 size=num_features, min_count=min_word_count,
                 window=window, sample=sample)
model.init_sims(replace=True)

model.save(os.path.join('.', 'models', model_name))
print(model.doesnt_match('man woman child kitchen'.split()))

