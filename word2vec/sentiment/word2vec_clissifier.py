# -*- coding: utf-8 -*-
# @bref :
import os
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from gensim.models.word2vec import Word2Vec

from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

def load_dataset(name, nrows=None):
    datasets = {
        'unlabeled_train': 'unlabeledTrainData.tsv',
        'labeled_train': 'labeledTrainData.tsv',
        'test': 'testData.tsv'
    }
    if name not in datasets:
        raise ValueError(name)
    data_file = os.path.join('.', 'data', datasets[name])
    df = pd.read_csv(data_file, sep='\t', escapechar='\\', nrows=nrows)#分隔符,转义字符
    print('Number of reviews:{}'.format(len(df)))
    return df

eng_stopwords = set(stopwords.words('english'))
def clean_text(text, remove_stopwords=False):
    text = BeautifulSoup(text,'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    if remove_stopwords:
        words = [w for w in words if w not in eng_stopwords]
    return words

model_name = '300features_40minwords_10window.model'
model = Word2Vec.load(os.path.join('models', model_name))

df = load_dataset('labeled_train')
df.head()

def to_review_vector(review):
    words = clean_text(review, remove_stopwords=True)
    array = np.array([model[w] for w in words if w in model])
    return pd.Series(array.mean(axis=0)) #axis=0表示计算行,1表示计算列

train_data_feature = df.review.apply(to_review_vector)
train_data_feature.head()

#用随机森林构建分类器
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest = forest.fit(train_data_feature, df.sentiment)

#在训练集上使用混淆矩阵测试
#confusion_matrix(df.sentiment, forest.predict(train_data_feature))

#清理占用内容的变量
del df
del train_data_feature

df = load_dataset('test')
print(df.head())
test_data_features = df.review.apply(to_review_vector)
test_data_features.head()

result = forest.predict(test_data_features)
output = pd.DataFrame({'id':df.id, 'sentiment':result})
output.to_csv(os.path.join('.', 'data', 'Word2Vec_model.csv'), index=False)
print(output.head())
del df
del test_data_features
del forest

# 对词向量进行聚类研究
word_vectors = model.syn0  #所有词的词向量
print word_vectors.shape[0]
print word_vectors.shape[1]
num_clusters = word_vectors.shape[0] // 10
kmeans_clustering = KMeans(n_clusters = num_clusters, n_jobs=4)
#使用kmeans对所有词进行聚类
idx = kmeans_clustering.fit_predict(word_vectors)
word_centroid_map = dict(zip(model.index2word, idx))
#pickle是数据持久存储模块
import pickle
filename = 'word_centroid_map_10avg.pickle'
with open(os.path.join('.', 'models', filename), 'wb') as f:
    pickle.dump(word_centroid_map, f)

for cluster in range(0, 10): #只打印10个
    print("\nCluster %d" % cluster)
    print([w for w,c in word_centroid_map.items() if c == cluster])

wordset = set(word_centroid_map.keys())
def make_cluster_bag(review):
    words = clean_text(review, remove_stopwords=True)
    return (pd.Series([word_centroid_map[w] for w in words if w in wordset])
            .value_counts()
            .reindex(range(num_clusters + 1), fill_value=0))

df = load_dataset('labeled_train')
df.head()

train_data_features = df.review.apply(make_cluster_bag)
print(train_data_features.head())

forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest = forest.fit(train_data_features, df.sentiment)
print confusion_matrix(df.sentiment, forest.predict(train_data_features))

del df
del train_data_features

df = load_dataset('test')
test_data_features = df.review.apply(make_cluster_bag)
print(test_data_features.head())

result = forest.predict(test_data_features)
output = pd.DataFrame({'id':df.id, 'sentiment':result})
output.to_csv(os.path.join('models', 'Word2Vec_BagOfClusters.csv'), index=False)
print output.head()

del df
del train_data_features
del forest














