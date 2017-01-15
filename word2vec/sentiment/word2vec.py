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
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    if remove_stopwords:
        words = [w for w in words if w not in eng_stopwords]
    return words

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
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

sentences = sum(df.review.apply(split_sentences), [])
#sentences = df.review.apply(split_sentences)
#print sentences

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


