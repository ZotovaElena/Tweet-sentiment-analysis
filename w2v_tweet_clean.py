#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 00:40:33 2018

@author: elena
"""

from gensim.models import Word2Vec
import pickle
import pandas as pd
import numpy as np
import re
from nltk import pos_tag, word_tokenize
import nltk
import string
import operator 
from collections import Counter
from collections import defaultdict
#import pymorphy2
#import progressbar
import pickle

emoticons_str = r"""
		(?:
			[:=;] # Eyes
			[oO\-]? # Nose (optional)
			[D\)\]\(\]/\\OpP] # Mouth
		)"""

regex_str = [
		emoticons_str,
		r'<[^>]+>', # HTML tags
		r'(?:@[\w_]+)', # @-mentions
		r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
		r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
	 
		r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
		r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
		r'(?:[\w_]+)', # other words
		r'(?:\S)' # anything else
	]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
    
def tokenize(s):
	return tokens_re.findall(s)

	 
def preprocess(s, lowercase=False):
	tokens = tokenize(s)
	if lowercase:
		tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
	return tokens



def is_number(s):
	number_list = ['0','1','2','3','4','5','6','7','8','9',',','.']
	number = True
	for c in s:
		if c not in number_list:
			number = False
	return number


def clean_tweets(tweets, model_path = None):
	
	#Get tweet list
	tweets = tweets.fillna('')
	tweets_text = tweets.text.values
	tweets_text = list(tweets.text.values)
	
    # Create stopword list
	stopwords = nltk.corpus.stopwords.words('russian')
	stopwords_delete = ['хорошо', 'лучше', 'нет', 'да', 'может', 'никогда', 'нельзя', 'всегда']
	stopwords_add = ['это', 'который']
			
	new_stopwords = []
	for word in stopwords:
		if word not in stopwords_delete:
			new_stopwords.append(word)
	stopwords = new_stopwords
	if len(stopwords_add) != 0:
		stopwords += stopwords_add
	punctuation = list(string.punctuation)
	punctuation += ['–', '—', '"']
	stop = stopwords + punctuation + ['rt', 'via']
    
    
    #Load words included in w2v model, if model exists
	model_vocab = None
	if model_path != None:
		model = Word2Vec.load(model_path)
		model_vocab = model.wv.vocab
	
    
	tweets_clean = []
	for tweet in tweets_text:
		tokens = preprocess(tweet, lowercase=True)
		tweet_tok = []
		for token in tokens:
			if token not in stop and not emoticon_re.search(token) and not is_number(token) and not token.startswith(('#', '@', 'http')):
				if model_vocab != None:
					if token in model_vocab:
						tweet_tok.append(token)
				else:
					tweet_tok.append(token)
										        
	#				lem = pymorphy2.MorphAnalyzer().parse(token)[0].normal_form 
	#				terms_only_all.append(lem)
		if len(tweet_tok) > 0:
			tweets_clean.append(tweet_tok)
	return tweets_clean


tweets_lems = pickle.load(open('tweets_lems.pkl', 'rb'))


model = Word2Vec.load('w2v_russ/ru.bin')
num_features = model.vector_size

#Clean tweet databases
negative = pd.read_csv('negative.csv', sep=',')
negative_clean = clean_tweets(negative, model_path='w2v_russ/ru.bin')
positive = pd.read_csv('positive.csv', sep=',')
positive_clean = clean_tweets(positive, model_path='w2v_russ/ru.bin')
elena = pd.read_csv('tweets.csv', sep=',')
elena_clean = clean_tweets(elena, model_path='w2v_russ/ru.bin')

#join positive and negative tweets and create labels
tweets_all = negative_clean + positive_clean
labels = np.array(['n']*len(negative_clean) + ['p']*len(positive_clean))

#save clean tweet databases and labels
dataDir = 'w2v_tweets_data/'
pickle.dump(tweets_all, open(dataDir+'tweets_clean2.pkl', 'wb'))
pickle.dump(labels, open(dataDir+'labels_tweets2.pkl', 'wb'))
pickle.dump(elena_clean, open(dataDir+'elena_clean2.pkl', 'wb'))








