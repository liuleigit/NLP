# -*- coding: utf-8 -*-
# @brief : 使用 bag of words + random forest进行评论的情感分析

import os
import re
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

#使用pandas读入训练数据
datafile = os.path.join('.', 'data', 'labeledTrainData.tsv')
df = pd.read_csv(datafile, sep='\t', escapechar='\\')
print('Number of reviews: {}'.format(len(df)))
print df.head()

def display(text, title):
    print(title)
    print("\n-------------我是分割线-------------\n")
    print(text)

raw_example = df['review'][1]
display(raw_example, '原始数据')
example = BeautifulSoup(raw_example, 'html.parser').get_text()

stopwords = {}.fromkeys([line.rstrip() for line in open('./stopwords.txt')])
eng_stopwords = set(stopwords)
def clean_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    words = [w for w in words if w not in eng_stopwords]
    return ' '.join(words)

print clean_text(raw_example)

df['clean_review'] = df.review.apply(clean_text)

vectorizer = CountVectorizer(max_features=5000)
train_data_features = vectorizer.fit_transform(df.clean_review).toarray()

forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data_features, df.sentiment)
confusion_matrix(df.sentiment, forest.predict(train_data_features))

datafile = os.path.join('.', 'data', 'testData.tsv')
df = pd.read_csv(datafile, sep='\t', escapechar='\\')
df['clean_review'] = df.review.apply(clean_text)
test_data_features = vectorizer.transform(df.clean_review).toarray()
result = forest.predict(test_data_features)
output = pd.DataFrame({'id':df.id, 'sentiment':result})
output.to_csv(os.path.join('.', 'data', 'Bag_of_Words_model.csv'), index=False)


