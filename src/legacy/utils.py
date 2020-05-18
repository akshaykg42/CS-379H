"""
Module that contains that contains a couple of utility functions
"""
import pickle
import nltk.data
import re
import string
from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path("../ner-model-2018.12.18.tar.gz")
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
tags = {
	'ALL': ['B-PER', 'I-PER', 'L-PER', 'B-ORG', 'I-ORG', 'L-ORG', 'B-LOC', 'I-LOC', 'L-LOC', 'B-MISC', 'I-MISC', 'L-MISC', 'U-PER', 'U-ORG', 'U-LOC', 'U-MISC'],
	'PER': ['B-PER', 'I-PER', 'L-PER', 'U-PER'],
	'ORG': ['B-ORG', 'I-ORG', 'L-ORG', 'U-ORG'],
	'LOC': ['B-LOC', 'I-LOC', 'L-LOC', 'U-LOC'],
	'MISC': ['B-MISC', 'I-MISC', 'L-MISC', 'U-MISC']
}
#pcr_documents = [' '.join(tokenizer.tokenize(doc)[len(pcr_oracles[i]):]) if re.search('\W\s*¶\s*\d\s*\W', doc) is None else doc for i, doc in enumerate(pcr_documents)]

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

def remove_named_entities(documents):
	out = []
	for document in tqdm(documents):
		new_doc = []
		for sentence in document:
			prediction = predictor.predict(sentence)
			replaced = [tag if tag != 'O' else word for i, (tag, word) in enumerate(list(zip(prediction["tags"], prediction["words"])))]
			i = 0
			reduced = []
			while(i < len(replaced)):
				word = replaced[i]
				if(word in tags['ALL']):
					tagtype = word[2:]
					while(i < len(replaced) and replaced[i] in tags[tagtype]):
						i += 1
					reduced.append(tagtype)
				else:
					reduced.append(word)
					i += 1
			new_doc.append(' '.join(reduced))
		out.append(new_doc)
	return out

def load(data_dir):
	with open(data_dir + '/raw/documents.pkl', 'rb') as f:
		documents = pickle.load(f)

	with open(data_dir + '/raw/summaries.pkl', 'rb') as f:
		summaries = pickle.load(f)

	with open(data_dir + '/raw/oracles.pkl', 'rb') as f:
		oracles = pickle.load(f)

	with open(data_dir + '/raw/types.pkl', 'rb') as f:
		types = pickle.load(f)

	return documents, summaries, oracles, types
