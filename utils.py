"""
Module that contains that contains a couple of utility functions
"""
import pickle
import nltk
import nltk.data
import re
import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer
from itertools import accumulate, permutations
from rouge import Rouge
from beam import *
rouge = Rouge()
rouge_type = 'rouge-2'
rouge_metric = 'f'
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stopwords = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's", "would"]

def optimize_oracle_indices(documents, summaries, oracles)
	for i in range(len(oracles)):
		summary = summaries[i]
		document = documents[i]
		oracle_indices = oracles[i]
		summary_sentences = tokenizer.tokenize(summary)
		document_sentences = tokenizer.tokenize(document)
		try:
			if(len(oracle_indices) <= 9):
				options = list(permutations(oracle_indices))
				best_option = options[0]
				best_score = sum([rouge.get_scores(summary_sentences[j], document_sentences[options[0][j]])[0][rouge_type][rouge_metric] for j in range(len(summary_sentences))])
				for option in options:
					score = sum([rouge.get_scores(summary_sentences[j], document_sentences[option[j]])[0][rouge_type][rouge_metric] for j in range(len(summary_sentences))])
					if(score > best_score):
						best_score = score
						best_option = option
				oracles[i] = best_option
		except ValueError:
			pass
	return oracles

def get_oracle_indices_and_rouge(documents, summaries):
	oracles = []
	rouge_scores = []
	for k in range(len(summaries)):
		document = documents[k]
		summary = summaries[k]
		document_sentences = tokenizer.tokenize(document)
		summary_sentences = tokenizer.tokenize(summary)
		beam = Beam(15)
		beam.add(('', []), 0)
		for i in range(len(summary_sentences)):
			new_beam = Beam(15)
			for curr in list(beam.get_elts_and_scores()):
				option, score = curr
				fragment, indices = option
				for j in range(len(summary_sentences), len(document_sentences)):
					if(j not in indices):
						new_fragment = fragment + ' ' + document_sentences[j]
						new_score = rouge.get_scores(new_fragment, ' '.join(summary_sentences[:i+1]))[0][rouge_type][rouge_metric]
						new_indices = [x for x in indices]
						new_indices.append(j)
						new_beam.add((new_fragment, new_indices), new_score)
			beam = new_beam
		oracle = list(beam.get_elts_and_scores())[0]
		option, score = oracle
		text, indices = option
		oracles.append(indices)
		rouge_scores.append(score)
	return oracles, rouge_scores

def get_rouge(hypothesis, reference, rougetype, scoretype):
	rougetype = 'rouge-' + rougetype
	return rouge.get_scores(hypothesis, reference)[0][rougetype][scoretype]

def pad_and_mask(batch_inputs):
	batch_size = len(batch_inputs)
	lengths = np.array([len(example) for example in batch_inputs])
	max_len = max(lengths)
	num_features = batch_inputs[0].shape[1]
	padded_inputs = np.zeros((batch_size, max_len, num_features))
	for i, example in enumerate(batch_inputs):
		for j, sentence in enumerate(example):
			padded_inputs[i][j] = sentence
	mask = np.arange(max_len) < lengths[:, None]
	padded_inputs = torch.from_numpy(padded_inputs).float().cuda()
	mask = (~(torch.from_numpy(mask).byte())).to(torch.bool).cuda()
	return mask, padded_inputs

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

# "Capped" at 64 i.e. vector length is 7 (for 1, 2, 4, 8, 16 , 32, 64+)
def bucketize_sent_lens(number):
	binary = [int(x) for x in bin(number)[2:]]
	if(len(binary) < 6):
		binary = [0] * 6 + binary
	big = 1 if any(binary[:-6]) else 0
	return [big] + binary[-6:]
	# If we just want high order bit then use this:
	'''
	index = min(int(math.log(number, 2)), 6)
	out = [0] * 7
	out[index] = 1
	return out[::-1]
	'''

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
			sent_len.extend([bucketize_sent_lens(len(nltk.word_tokenize(sent))) for sent in doc_sents])
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
	X = np.array([np.array(X[splits[i]:splits[i+1]]) for i in range(len(splits) - 1)])
	y = np.array(y)
	return X, y

def load(documents, summaries, oracle_indices):
	with open(documents, 'rb') as f:
		documents = pickle.load(f)

	with open(summaries, 'rb') as f:
		summaries = pickle.load(f)

	with open(oracle_indices, 'rb') as f:
		oracles = pickle.load(f)

	return documents, summaries, oracles
