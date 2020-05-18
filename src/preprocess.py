import os
import re
import json
import glob
import nltk
import torch
import spacy
import string
import argparse
import nltk.data
import numpy as np
from beam import *
from rouge import Rouge
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from transformers import BertTokenizer
from itertools import accumulate, permutations
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

rouge = Rouge()
rouge_metric = 'f'
rouge_type = 'rouge-1'

sp = spacy.load('en_core_web_sm')
word_tokenizer = nltk.word_tokenize
sentence_tokenizer = nltk.sent_tokenize
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def load_doc(filename):
	file = open(filename, encoding='utf-8')
	text = file.read()
	file.close()
	return text

def split_doc(doc):
	index = doc.find('@highlight')
	document, summary = doc[:index], doc[index:].split('@highlight')
	document = sentence_tokenizer(document)
	summary = [h.strip() for h in summary if len(h) > 0]
	return document, summary

def load_data(directory):
	data = list()
	files = list()
	for name in os.listdir(directory):
		try:
			filename = directory + '/' + name
			datum = load_doc(filename)
			document, summary = split_doc(datum)
			data.append({'document':document, 'summary':summary})
			files.append(filename)
		except UnicodeDecodeError:
			print(name)
	return data, files

def create_test_train_val_split(bert_data, linear_data):
	indices_train, indices_test = train_test_split(list(range(len(bert_data))), test_size=0.2)
	indices_train, indices_val = train_test_split(indices_train, test_size=0.25)

	bert_train = [bert_data[i] for i in indices_train]
	bert_test = [bert_data[i] for i in indices_test]
	bert_val = [bert_data[i] for i in indices_val]

	linear_train = [linear_data[i] for i in indices_train]
	linear_test = [linear_data[i] for i in indices_test]
	linear_val = [linear_data[i] for i in indices_val]

	return (bert_train, bert_test, bert_val), (linear_train, linear_test, linear_val), (indices_train, indices_test, indices_val)

def jsonify(raw_path, dataset_name):
	data, files = load_data(raw_path)
	documents = [datum['document'] for datum in data]
	summaries = [datum['summary'] for datum in data]

	write_dir = '../data/{}/raw/'.format(dataset_name)
	if not os.path.exists(write_dir):
		os.makedirs(write_dir)

	with open(write_dir + 'documents.json', 'w') as outfile:
		json.dump(documents, outfile)
	with open(write_dir + 'summaries.json', 'w') as outfile:
		json.dump(summaries, outfile)

def preprocess_sentence(sentence):
	# Remove non alphanumeric/space
	sentence = re.sub(r'[^a-zA-Z\d\s]', ' ', sentence)
	# Replace multispace with single space
	sentence = re.sub(r'\s+', ' ', sentence, flags=re.I)
	# Convert to lowercase
	sentence = sentence.lower()

	return sentence
	# tokenized = word_tokenizer(sentence)
	# return ' '.join(tokenized)

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

def get_linear_features(documents):
	preprocessed_documents_flat, feat_pos, feat_len, doc_lens = [], [], [], []
	doc_lens.append(0)

	for i, document in enumerate(documents):
		for sentence in document:
			preprocessed_sentence = preprocess_sentence(sentence)
			preprocessed_documents_flat.append(preprocessed_sentence)
			feat_len.append(
				bucketize_sent_lens(
					len(word_tokenizer(sentence))
				)
			)
			feat_pos.append([(i+1)/len(documents)])
		doc_lens.append(len(document))

	# Converting preprocessed sentences to features
	vectorizer = CountVectorizer(max_features=10000, min_df=5,
							max_df=0.99, stop_words=stopwords.words('english'),
							ngram_range=(1, 2))

	feats_bow = vectorizer.fit_transform(preprocessed_documents_flat).toarray()

	# Adding features for sentence length and position
	features = np.append(feats_bow, feat_len, axis=1)
	features = np.append(features, feat_pos, axis=1)

	# Separating into documents again for training
	splits = list(accumulate(doc_lens))
	features = [features[splits[i]:splits[i+1]] for i in range(len(splits) - 1)]
	features = [torch.tensor(f) for f in features]

	return features

def tokenize_bert(documents):
	tokenized_documents = []
	for i, document in enumerate(documents):
		tokenized_sentences = []
		for sentence in document:
			encoded_sent = bert_tokenizer.encode(
				sentence,
				add_special_tokens = True,
				max_length = 512,
				return_tensors = 'pt'
			)
			tokenized_sentences.append(encoded_sent)
		tokenized_documents.append(tokenized_sentences)
	return tokenized_documents

def bert_tokens_and_linear_features(dataset_name, overwrite):
	folders = ['linear/', 'bert/']
	subfolders = ['train/', 'test/', 'val/']
	json_path = '../data/{}/raw/documents.json'.format(dataset_name)
	with open(json_path) as json_file:
		documents = json.load(json_file)

	dataset_path = '../data/{}/'.format(dataset_name)
	for folder in folders:
		for subfolder in subfolders:
			path = dataset_path + folder + subfolder
			if not os.path.exists(path):
				os.makedirs(path)
			files = glob.glob(path + '*')
			if(not overwrite and files):
				print('Linear features and BERT tokens for this dataset have already been extracted, and a test-train-val split has been created. If you would like to overwrite these and generate a new test-train-val split, or if you have new data, please run again with the -overwrite flag.')
				return
			for file in files:
				os.remove(file)

	bert_tokens = tokenize_bert(documents)
	linear_features = get_linear_features(documents)

	bert_data, linear_data, indices = create_test_train_val_split(bert_tokens, linear_features)

	for subfolder, subindices, linear_sub, bert_sub in list(zip(subfolders, indices, linear_data, bert_data)):
		for i, index in enumerate(subindices):
			torch.save(linear_sub[i], dataset_path + 'linear/' + subfolder + str(index) + '.pt')
			torch.save(bert_sub[i], dataset_path + 'bert/' + subfolder + str(index) + '.pt')

	with open(dataset_path + 'train_test_split.txt', 'w') as outfile:
		train_indices, test_indices, val_indices = indices
		outfile.write(str(train_indices) + '\n' + str(test_indices) + '\n' + str(val_indices))

def get_rouge(hypothesis, reference, rougetype, scoretype):
	return rouge.get_scores(hypothesis, reference)[0][rougetype][scoretype]

def get_vanilla_oracles(documents, summaries):
	oracles = []
	for document, summary in list(zip(documents, summaries)):
		document_sentences = [' '.join([word.lemma_ for word in sp(sentence)]) for sentence in document]
		summary_sentences = [' '.join([word.lemma_ for word in sp(sentence)]) for sentence in summary]
		oracle = []
		for summary_sentence in summary_sentences:
			best_score = -1.0
			oracle_sentence = -1
			for j in range(len(document_sentences)):
				score = get_rouge(summary_sentence, document_sentences[j], rouge_type, rouge_metric)
				if(score > best_score):
					oracle_sentence = j
					best_score = score
			oracle.append(oracle_sentence)
		oracles.append(oracle)
	return oracles

def optimize_beam_oracles(documents, summaries, oracles):
	optimized_oracles = []
	for document, summary, oracle_indices in list(zip(documents, summaries, oracles)):
		best_option = oracle_indices
		if(len(oracle_indices) <= 9):
			options = list(permutations(oracle_indices))
			best_score = sum([get_rouge(summary[i], document[best_option[i]], rouge_type, rouge_metric) for i in range(len(summary))])
			for option in options:
				score = sum([get_rouge(summary[i], document[option[i]], rouge_type, rouge_metric) for i in range(len(summary))])
				if(score > best_score):
					best_score = score
					best_option = option
		optimized_oracles.append(best_option)
	return optimized_oracles

#todo: functionality to control allowing duplicate sentences?
def get_beam_oracles(documents, summaries):
	oracles = []
	for document, summary in list(zip(documents, summaries)):
		document = [' '.join([word.lemma_ for word in sp(sentence)]) for sentence in document]
		summary = [' '.join([word.lemma_ for word in sp(sentence)]) for sentence in summary]
		beam = Beam(15)
		beam.add(('', []), 0)
		for i in range(len(summary)):
			new_beam = Beam(15)
			for curr in list(beam.get_elts_and_scores()):
				option, score = curr
				fragment, indices = option
				for j in range(len(document)):
					if(j not in indices):
						new_fragment = fragment + ' ' + document[j]
						new_score = get_rouge(new_fragment, ' '.join(summary[:i+1]), rouge_type, rouge_metric)
						new_beam.add((new_fragment, indices + [j]), new_score)
			beam = new_beam
		oracle = list(beam.get_elts_and_scores())[0]
		option, score = oracle
		text, indices = option
		oracles.append(indices)
	return oracles

def construct_oracles(dataset_name, vanilla_oracles):
	json_path = '../data/{}/raw/'.format(dataset_name)
	with open(json_path + 'documents.json') as json_file:
		documents = json.load(json_file)
	with open(json_path + 'summaries.json') as json_file:
		summaries = json.load(json_file)

	if(vanilla_oracles):
		oracles = get_vanilla_oracles(documents, summaries)
	else:
		oracles = get_beam_oracles(documents, summaries)
		oracles = optimize_beam_oracles(documents, summaries, oracles)

	with open(json_path + 'oracles.json', 'w') as outfile:
		json.dump(oracles, outfile)

def get_topic_representations(dataset_name):
	json_path = '../data/{}/raw/summaries.json'.format(dataset_name)
	with open(json_path) as json_file:
		summaries = json.load(json_file)

	summaries_flat = []
	for summary in summaries:
		summaries_flat.extend(summary)

	tfidf_vectorizer = TfidfVectorizer(max_df=0.95, max_features=10000,
                                 min_df=2, stop_words=stopwords.words('english'),
                                 use_idf=True, lowercase=True,
                                 ngram_range=(1,3))

	tfidf_matrix = tfidf_vectorizer.fit_transform(summaries_flat)

	#TODO: Make interface for users to seed clusters - these values are hardcoded for earthquakes
	topic_samples = [[4, 12, 1012, 1014, 897, 795, 770, 1408, 754, 808, 819, 900, 909, 952, 1215],[1423, 1521, 1633, 1649, 1705, 1821, 1909, 49, 171],[377, 401, 512, 536, 624, 889, 1396, 1518, 552, 1312, 592],[839, 892, 1170, 1175, 1242, 1279, 1287, 1314, 1458, 1482, 1642, 1680, 1475, 1670, 1691, 1889],[1065, 1234, 1372, 1532, 1872, 334, 1147, 1253, 1491, 1679],[1822, 1846, 1511, 1631, 1838, 1910, 41, 348, 388, 415, 546, 646, 670, 419, 787, 1560, 1704, 138, 9, 81],[686, 674, 721, 783, 872, 874, 1124, 1132, 1138, 1194, 1332, 1338, 1340, 167, 237, 427]]
	num_topics = len(topic_samples)

	topic_seed_centroids = []
	for topic_sample in topic_samples:
		topic_vectors = [tfidf_matrix[index] for index in topic_sample]
		topic_vectors = np.array([vector.toarray() for vector in topic_vectors])
		topic_vectors = topic_vectors.reshape(topic_vectors.shape[0], topic_vectors.shape[2])
		km = KMeans(n_clusters=1)
		km.fit(topic_vectors)
		seed_centroid = km.cluster_centers_[0]
		topic_seed_centroids.append(seed_centroid)

	km = KMeans(n_clusters=num_topics, init=np.array(topic_seed_centroids))
	km.fit(tfidf_matrix)

	sentences_to_topics = {}
	sentences_by_topic = {}

	for topic in range(num_topics):
		indices = [index for index, label in enumerate(km.labels_) if label == topic]
		topic_sentences = []
		for index in indices:
			sentences_to_topics[summaries_flat[index]] = topic
			topic_sentences.append(summaries_flat[index])
		sentences_by_topic[topic] = topic_sentences

	# topic_list_path = '../data/' + dataset_name + '/'
	# for topic in sentences_by_topic:
	# 	with open(topic_list_path + 'topic {} sentences.txt'.format(topic), 'w') as f:
	# 		f.write('\n\n'.join(sentences_by_topic[topic]))

	topic_representations = [[[sentences_to_topics[sentence]] for sentence in summary] for summary in summaries]

	write_dir = '../data/{}/raw/'.format(dataset_name)
	with open(write_dir + 'topics.json', 'w') as outfile:
		json.dump(topic_representations, outfile)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-raw_path', default='')
	parser.add_argument('-dataset_name')
	parser.add_argument('-mode')
	parser.add_argument('-overwrite', action='store_true', default=False)
	parser.add_argument('-vanilla_oracles', action='store_true', default=False)
	args = parser.parse_args()

	raw_path, dataset_name, mode, overwrite, vanilla_oracles =\
		args.raw_path, args.dataset_name, args.mode, args.overwrite, args.vanilla_oracles

	if(args.mode == 'jsonify'):
		jsonify(raw_path, dataset_name)
	elif(args.mode == 'construct_oracles'):
		construct_oracles(dataset_name, vanilla_oracles)
	elif(args.mode == 'topic_clustering'):
		get_topic_representations(dataset_name)
	elif(args.mode == 'bert_tokens_and_linear_features'):
		bert_tokens_and_linear_features(dataset_name, overwrite)
