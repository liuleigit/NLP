# -*- coding: utf-8 -*-
# @bref :
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import jieba
from sklearn.externals import joblib
from sklearn.svm import SVC
import sys
reload(sys)
sys.setdefaultencoding('utf8')

def load_file_and_preprocessing():
    neg = pd.read_excel('data/neg.xls', header=None, index=None)
    pos = pd.read_excel('data/pos.xls', header=None, index=None)

    cw = lambda x: list(jieba.cut(x)) # cut的返回值类型是generator
    pos['word'] = pos[0].apply(cw)
    neg['word'] = neg[0].apply(cw)
    #使用1做positive sentiment, 0 for negative sentiment
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))
    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos['words'], neg['words'])), y, test_size=0.2)
    np.save('svm_data/y_train.npy', y_train)
    np.save('svm_data/y_test.npy', y_test)
    return x_train, x_test

def build_sentence_vector(text, size, imdb_w2v):
    #reshape用来修改np.array形状。(1, size)表示2维,大小分别是1和size
    #a=np.array([1,2,3,4,5,6,7,8])
    #a.reshape((2,2,2)) ===> array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count  #无论vec几维, 每个元素除以count
    return vec

def get_train_vecs(x_train, x_test):
    n_dim = 300
    imdb_w2v = Word2Vec(size=n_dim, min_count=10)
    #词表
    imdb_w2v.build_vocab(x_train)
    #在评论训练集上建模
    imdb_w2v.train(x_train)

    train_vecs = np.concatenate([build_sentence_vector(z, n_dim, imdb_w2v) for z in x_train])
    np.save('svm_data/train_vecs.npy', train_vecs)
    print train_vecs.shape

    #在测试集上训练
    imdb_w2v.train(x_test)
    imdb_w2v.save('svm_data/w2v_model/w2v_model.pkl')
    test_vecs = np.concatenate([build_sentence_vector(z, n_dim, imdb_w2v) for z in x_test])
    np.save('svm_data/test_vecs.npy', test_vecs)
    print test_vecs.shape

def get_data():
    train_vecs = np.load('svm_data/train_vecs.npy')
    y_train = np.load('svm_data/y_train.npy')
    test_vecs = np.load('svm_data/test_vecs.npy')
    y_test = np.load('svm_data/y_test.npy')
    return train_vecs, y_train, test_vecs, y_test

def svm_train(train_vecs, y_train, test_vecs, y_test):
    clf = SVC(kernel='rbf', verbose=True)
    clf.fit(train_vecs, y_train)
    joblib.dump(clf, 'svm_data/svm_model/model.pkl')
    print clf.score(test_vecs, y_test)

def get_predict_vecs(words):
    n_dim = 300
    #使用了test数据训练出来的词向量
    imdb_w2v = Word2Vec.load('svm_data/w2v_model/w2v_model.pkl')
    train_vecs = build_sentence_vector(words, n_dim, imdb_w2v)
    return train_vecs

def svm_predict(string):
    words = jieba.cut(string)
    words_vecs = get_predict_vecs(words)
    clf = joblib.load('svm_data/svm_model/model.pkl')
    result = clf.predict(words_vecs)
    if int(result[0]) == 1:
        print string, ' positive'
    else:
        print string, ' negative'


x_train, x_test = load_file_and_preprocessing()
get_train_vecs(x_train, x_test)
train_vecs, y_train, test_vecs, y_test = get_data()
svm_train(train_vecs, y_train, test_vecs, y_test)

string = '电池充完了电连手机都打不开,简直烂的要命, 真是金玉其外,败絮其中! 连5号电池都不如'
svm_predict(string)


