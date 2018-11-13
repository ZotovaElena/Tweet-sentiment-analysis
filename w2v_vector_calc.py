#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 00:40:33 2018

@author: elena
"""

import pickle
from gensim.models import Word2Vec
import numpy as np

def featureVecMethod(words, model, num_features):
    # Pre-initialising empty numpy array for speed
    featureVec = np.zeros(num_features,dtype="float32")
    nwords = 0
    
    #Converting Index2Word which is a list to a set for better speed in the execution.
    index2word_set = set(model.wv.index2word)
    
    for word in  words:
        if word in index2word_set:
            nwords = nwords + 1
#            featureVec = np.add(featureVec,model[word])
			featureVec = featureVec + model[word]
    
    # Dividing the result by number of words to get average
#    featureVec = np.divide(featureVec, nwords)
	featureVec = featureVec/nwords
    return featureVec


def getAvgFeatureVecs(tweets, model, num_features):
    counter = 0
    tweetFeatureVecs = np.zeros((len(tweets),num_features),dtype="float32")
#    all_tweets = len(tweets)
    for i, tweet in enumerate(tweets):
#        print(i,'from',all_tweets)
        # Printing a status message every 1000th review
        if counter%1000 == 0:
            print("Review %d of %d"%(counter,len(tweets)))
            
        tweetFeatureVecs[counter] = featureVecMethod(tweet, model, num_features)
        counter = counter+1
        
    return tweetFeatureVecs

dataDir = 'w2v_tweets_data/'
tweets_clean = pickle.load(open(dataDir+'tweets_clean.pkl', 'rb'))
elena_clean = pickle.load(open(dataDir+'elena_clean.pkl', 'rb'))

model = Word2Vec.load('w2v_russ/ru.bin')
num_features = model.vector_size

#Get average w2v per tweet
X_tweets = getAvgFeatureVecs(tweets_clean, model, num_features)
X_elena = getAvgFeatureVecs(elena_clean, model, num_features)

pickle.dump(X_tweets, open(dataDir+'tweets_w2v.pkl', 'wb'))
pickle.dump(X_elena, open(dataDir+'elena_w2v.pkl', 'wb'))