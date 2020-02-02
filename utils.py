"""
Module that contains that contains a couple of utility functions
"""

import pickle
import nltk
import nltk.data
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def preprocess_sentence(sentence):
	# Remove special chars
	sentence = re.sub(r'\W', ' ', sentence)
	# Remove single chars
	sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)
	# Remove single chars from start
	sentence = re.sub(r'\^[a-zA-Z]\s+', ' ', sentence)
	# Replace multispace with single space
	sentence = re.sub(r'\s+', ' ', sentence, flags=re.I)
	# Remove prefixed b'
	sentence = re.sub(r'^b\s+', '', sentence)
	# Convert to lowercase
	sentence = sentence.lower()
	
	tokenized = nltk.word_tokenize(sentence)
	return ' '.join(tokenized)

def get_features_and_labels(pcr_documents, pcr_summaries, pcr_oracles, type):
	X = []
	sent_pos = []
	sent_len = []
	doc_lens = [0]
	y = []
	indexerror = []
	for i in range(len(pcr_documents)):
		doc_sents = tokenizer.tokenize(pcr_documents[i])
		try:
			positive_index = pcr_oracles[i][type]
			y_i = [0] * len(doc_sents)
			y_i[positive_index] = 1
			X_i = [preprocess_sentence(sent) for sent in doc_sents]
			y.append(positive_index)
			X.extend(X_i)
			sent_len.extend([[len(nltk.word_tokenize(sent))] for sent in doc_sents])
			sent_pos.extend([[(j+1) / len(doc_sents)] for j in range(len(doc_sents))])
			doc_lens.append(len(doc_sents))
		except IndexError:
			indexerror.append(i)
	# Converting preprocessed sentences to features
	vectorizer = CountVectorizer(max_features=5000, min_df=4, max_df=0.99, stop_words=stopwords)
	X = vectorizer.fit_transform(X).toarray()
	# Adding features for sentence length and position
	X = np.append(X, sent_len, axis=1)
	X = np.append(X, sent_pos, axis=1)
	# Separating into documents again for training
	splits = list(accumulate(doc_lens))
	X = np.array([X[splits[i]:splits[i+1]] for i in range(len(splits) - 1)])
	y = np.array(y)
	return X, y, indexerror

def load(documents, summaries, oracle_indices, sent_type):
	
	"""
	Loads the data that is provided
	@param filename: The name of the data file. Can be either 'tux_train.dat' or 'tux_val.dat'
	@return images: Numpy array of all images where the shape of each image will be W*H*3
	@return labels: Array of integer labels for each corresponding image in images
	"""

	with open(documents, 'rb') as f:
		pcr_documents = pickle.load(f)

	with open(summaries, 'rb') as f:
		pcr_summaries = pickle.load(f)

	with open(oracle_indices, 'rb') as f:
		pcr_oracles = pickle.load(f)

	return get_features_and_labels(documents, summaries, oracles, sent_type)