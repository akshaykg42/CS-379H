import argparse
import torch
import os
import json
import string
import glob
import nltk
import nltk.data
import numpy as np
from nltk.corpus import stopwords
from rouge import Rouge
import re
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from itertools import accumulate, permutations
from transformers import BertTokenizer

rouge = Rouge()
rouge_type = 'rouge-1'
rouge_metric = 'f'

sp = spacy.load('en_core_web_sm')
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = nltk.word_tokenize
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
	write_dir = '../data/' + dataset_name + '/raw/'
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
	vectorizer = CountVectorizer(max_features=10000, min_df=5, max_df=0.99, stop_words=stopwords.words('english'), ngram_range=(1, 2))
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
	folders = ['linear_features/', 'bert_tokens/']
	subfolders = ['train/', 'test/', 'val/']
	json_path = '../data/' + dataset_name + '/raw/documents.json'
	with open(json_path) as json_file:
		documents = json.load(json_file)

	dataset_path = '../data/' + dataset_name + '/'
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
			torch.save(linear_sub[i], dataset_path + 'linear_features/' + subfolder + str(index) + '.pt')
			torch.save(bert_sub[i], dataset_path + 'bert_tokens/' + subfolder + str(index) + '.pt')

	with open(dataset_path + 'train_test_split.txt', 'w') as outfile:
		train_indices, test_indices, val_indices = indices
		outfile.write(str(train_indices) + '\n' + str(test_indices) + '\n' + str(val_indices))

def get_rouge(hypothesis, reference, rougetype, scoretype):
	rougetype = 'rouge-' + rougetype
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
			for option in options
				score = sum([get_rouge(summary[i], document[option[i]], rouge_type, rouge_metric) for i in range(len(summary))])
				if(score > best_score):
					best_score = score
					best_option = option
		optimized_oracles.append(best_option)
	return optimized_oracles

def get_beam_oracles(documents, summaries):
	oracles = []
	rouge_scores = []
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
				for j in range(len(summary), len(document)):
					if(j not in indices):
						new_fragment = fragment + ' ' + document[j]
						new_score = get_rouge(new_fragment, ' '.join(summary[:i+1]), rouge_type, rouge_metric)
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

def construct_oracles(dataset_name, vanilla_oracles):
	json_path = '../data/' + dataset_name + '/raw/'
	with open(json_path) as json_file:
		documents = json.load(json_file + 'documents.json')
		summaries = json.load(json_file + 'summaries.json')

	if(vanilla_oracles):
		oracles = get_vanilla_oracles(documents, summaries)
	else:
		oracles = get_beam_oracles(documents, summaries)
		oracles = optimize_beam_oracles(documents, summaries, oracles)

	with open(write_dir + 'oracles.json', 'w') as outfile:
		json.dump(oracles, outfile)

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
		pass
	elif(args.mode == 'bert_tokens_and_linear_features'):
		bert_tokens_and_linear_features(dataset_name, overwrite)












