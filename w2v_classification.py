# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 15:12:16 2018

@author: zotov
"""


'Test original'

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn import preprocessing
from sklearn.metrics import recall_score

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.mixture import GaussianMixture

import numpy as np

import pickle
from gensim.models import Word2Vec

dataDir = 'w2v_tweets_data/'
tweets_w2v = pickle.load(open(dataDir+'tweets_w2v.pkl', 'rb'))
labels = pickle.load(open(dataDir+'labels_tweets.pkl', 'rb'))
elenaw2v = pickle.load(open(dataDir+'elena_w2v.pkl', 'rb'))

#Encode the labes to int
le = preprocessing.LabelEncoder()
le.fit(['p','n'])
labels_int = le.transform(labels)

tweets_train, tweets_test, y_train, y_test = train_test_split(tweets_w2v, labels_int, test_size=0.25, random_state=0)

tweets_train = tweets_train.astype(np.float64)
tweets_test = tweets_test.astype(np.float64)
y_train = y_train.astype(np.float64)
y_test = y_test.astype(np.float64)

n_train = 5000
n_test = 1000

tweets_train = tweets_train[:n_train]
y_train = y_train[:n_train]
tweets_test = tweets_test[:n_test]
y_test = y_test[:n_test]

names = ["Naive Bayes", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Neural Net", "AdaBoost",
         "QDA"]

classifiers = [
    GaussianNB(),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), n_jobs=4),
    DecisionTreeClassifier(max_depth=5),
    MLPClassifier(alpha=1, verbose=True),
    AdaBoostClassifier(),
    QuadraticDiscriminantAnalysis()]

for name, clf in zip(names, classifiers):
    clf.fit(tweets_train, y_train)
    y_pred = clf.predict(tweets_test)
    
    f_score_macro = f1_score(y_test, y_pred, average='macro')
    f_score_micro = f1_score(y_test, y_pred, average='micro')
    precision = precision_score(y_test, y_pred, average='macro') 
    recall = recall_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(name, 'f-score macro:', f_score_macro)
    print(name, 'f-score micro:', f_score_micro)
    print(name, 'precision: ', precision)
    print(name, 'recall: ', recall)
    

n_gaussians = 256

GMM_p = GaussianMixture(n_components=n_gaussians, covariance_type='diag', verbose=2)
GMM_n = GaussianMixture(n_components=n_gaussians, covariance_type='diag', verbose=2)

GMM_p.fit(tweets_train[y_train==1])
GMM_n.fit(tweets_train[y_train==0])

scores_p = GMM_p.score_samples(tweets_test)
scores_n = GMM_n.score_samples(tweets_test)

y_pred = np.array([])

for s_p, s_n in zip(scores_p, scores_n):
    if s_p > s_n:
        y_pred = np.append(y_pred, 1) 
    else:
        y_pred = np.append(y_pred, 0)
score = f1_score(y_test, y_pred, average='macro')
print(score)



