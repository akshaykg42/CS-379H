"""
Module that contains that contains a couple of utility functions
"""
import pickle
import nltk
import nltk.data
from nltk.corpus import stopwords
import re
import string
import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer
from itertools import accumulate, permutations
from rouge import Rouge
from beam import *
rouge = Rouge()
rouge_type = 'rouge-1'
rouge_metric = 'r'
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#stopwords = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's", "would"]
#pcr_documents = [' '.join(tokenizer.tokenize(doc)[len(pcr_oracles[i]):]) if re.search('\W\s*¶\s*\d\s*\W', doc) is None else doc for i, doc in enumerate(pcr_documents)]


def get_vanilla_oracles(documents, summaries):
	oracles = []
	for document, summary in list(zip(documents, summaries)):
		document_sentences = tokenizer.tokenize(document)
		summary_sentences = tokenizer.tokenize(summary)
		oracle = []
		for summary_sentence in summary_sentences:
			best_score = -1.0
			oracle_sentence = -1
			for j in range(len(summary_sentences), len(document_sentences)):
				score = rouge.get_scores(summary_sentence, document_sentences[j])[0][rouge_type][rouge_metric]
				if(score > best_score):
					oracle_sentence = j
					best_score = score
			oracle.append(oracle_sentence)
		oracles.append(oracle)
	return oracles

def optimize_beam_oracles(documents, summaries, oracles):
	for document, summary, oracle_indices in list(zip(documents, summaries, oracles)):
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

def get_beam_oracles(documents, summaries):
	oracles = []
	rouge_scores = []
	for document, summary in list(zip(documents, summaries)):
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
	return oracles

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

def get_features_and_labels(pcr_documents, pcr_summaries, pcr_oracles, sentence_type):
	X = []
	sent_pos = []
	sent_len = []
	doc_lens = [0]
	y = []
	indexerror = []
	for i in range(len(pcr_documents)):
		doc_sents = tokenizer.tokenize(pcr_documents[i])
		try:
			positive_index = pcr_oracles[i][sentence_type]
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
	vectorizer = CountVectorizer(max_features=7500, min_df=5, max_df=0.99, stop_words=stopwords.words('english'))
	X = vectorizer.fit_transform(X).toarray()
	# Adding features for sentence length and position
	X = np.append(X, sent_len, axis=1)
	X = np.append(X, sent_pos, axis=1)
	# Separating into documents again for training
	splits = list(accumulate(doc_lens))
	X = np.array([np.array(X[splits[i]:splits[i+1]]) for i in range(len(splits) - 1)])
	y = np.array(y)
	return X, y

def clean_document(document):
	document = document.replace('Crim.', 'Criminal')
	document = document.replace('No.', 'Number')
	document = document.replace('Nos.', 'Numbers')
	document = document.replace('App.', 'Appeal')
	document = document.replace('Tenn.', 'Tennessee')
	document = document[re.search('\W\s*¶\s*1\s*\W', document).end():] if re.search('\W\s*¶\s*1\s*\W', document) is not None else document
	lines = tokenizer.tokenize(document)
	lines[0] = lines[0][re.search('[a-zA-Z]\d+CCA-[a-zA-Z0-9]{2}-[a-zA-Z0-9]{1,3}', lines[0]).end():] if re.search('[a-zA-Z]\d+CCA-[a-zA-Z0-9]{2}-[a-zA-Z0-9]{1,3}', lines[0]) != None else lines[0]
	cleaned = list()
	# prepare a translation table to remove punctuation
	table = str.maketrans('', '', string.punctuation)
	lines = [line[re.search('OPINION', line).start()+7:] if re.search('OPINION', line) != None else line for line in lines]
	lines = [re.sub('-\s*[0-9]*\s*-', '', line) for line in lines]
	lines = [re.sub('__+', '', line) for line in lines]
	lines = [' '.join(line.split()) for line in lines]
	for line in lines:
		# tokenize on white space
		line = line.split()
		# convert to lower case
		line = [word.lower() for word in line]
		# remove punctuation from each token
		line = [w.translate(table) for w in line]
		# remove tokens with numbers in them
		line = [word for word in line if word.isalpha()]
		# store as string
		cleaned.append(' '.join(line))
	# remove empty strings
	indices_to_keep = [i for i in range(len(lines)) if len(cleaned[i].split()) > 5 or 'affirm' in cleaned[i]]
	return ' '.join([lines[i] for i in indices_to_keep])

def load(documents, summaries, oracle_indices):
	with open(documents, 'rb') as f:
		documents = pickle.load(f)

	with open(summaries, 'rb') as f:
		summaries = pickle.load(f)

	with open(oracle_indices, 'rb') as f:
		oracles = pickle.load(f)

	return documents, summaries, oracles
