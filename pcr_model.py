import pickle
import nltk
import nltk.data
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from itertools import accumulate
import numpy as np

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stopwords = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's", "would"]

with open('pcr_oracles.pkl', 'rb') as f:
	pcr_oracles = pickle.load(f)

with open('pcr_documents.pkl', 'rb') as f:
	pcr_documents = pickle.load(f)

with open('pcr_summaries.pkl', 'rb') as f:
	pcr_summaries = pickle.load(f)

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
	'''
	tagged = nltk.pos_tag(tokenized)
	unstemmed = [pair[0] for pair in tagged if pair[1].startswith('VB')]
	#s = Sentence(sentence)
	#tagger.predict(s)
	#d = s.to_dict(tag_type='pos')
	#entities = d['entities']
	#unstemmed = [entity['text'] for entity in entities if entity['type'] == 'VERB']
	stop_removed = [word for word in unstemmed if word not in stopwords]
	return ' '.join([porter.stem(word) for word in stop_removed])
	'''

# Takes in documents, summaries, oracle indices and type (oracle sentence #) and returns X, y (X is each document represented as group of sentence features based on bag-of-words and the length/position of sentence)
def get_x_and_y(pcr_documents, pcr_summaries, pcr_oracles, type):
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

'''
MODEL STUFF BELOW
'''

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

#X consists of n examples, each example is a document containing some sentences represented by fixed number of features
def compute_cost(X, y, theta):
	# number of examples
	n = len(y)
	epsilon = 0 #1e-5
	# Get score for each sentence in each document by @ with params
	scored = np.array([x @ theta for x in X])
	# Softmax over scores for each document to obtain prob distribution for each document
	softmaxed = [softmax(scores) for scores in scored]
	# y[i] is the index of oracle extracted sentence for example i, need - log(softmax prob for this sentence) as cost for example i
	h = [-np.log(softmaxed[i][y[i]]) + epsilon for i in range(n)]
	# Take avg across all examples and return
	cost = (1/n)*np.sum(h)
	return cost

def predict(X, params):
	# For a single example, simply @ with params to obtain scores, softmax to obtain probabilities, use argmax to return best index
	return np.argmax(softmax(X @ params))

def gradient_descent(X, y, params, learning_rate, iterations):
	# Number of examples
	n = len(y)
	# Keep track of cost
	cost_history = np.zeros((iterations,1))
	# Need to flatten examples since update needs to touch params on every sentence, not just every document
	Xflat = np.vstack(X)
	# Need one-hot label for each sentence in every document (1 if sentence is oracle extracted, 0 otherwise)
	y_onehot = np.array([np.zeros(len(x)) for x in X])
	for i in range(n):
		y_onehot[i][y[i]] = 1
	# Iterate
	for i in range(iterations):
		# Get softmax scores for each sentence in each document
		softmaxs = np.array([np.squeeze(softmax(x @ params)) for x in X])
		# For each sentence, multiply features by [score - label] (label is 0 or 1 taken from onehot) and use the avg of the results across all examples to update params
		params = params - \
			(learning_rate/n) * \
			(
				Xflat.T @ np.hstack(np.squeeze(softmaxs) - y_onehot) # features X [score - label]
			).reshape((len(params), 1))
		# Update cost history
		cost_history[i] = compute_cost(X, y, params)
		print('Cost at end of iteration {} is {}'.format(i, cost_history[i]))
	return (cost_history, params)

def train(X, y):
	num_features = np.size(X[0],1)
	params = np.zeros((num_features,1))
	iterations = 1500
	learning_rate = 0.03
	initial_cost = compute_cost(X, y, params)
	print("Initial Cost is: {}\n".format(initial_cost))
	(cost_history, params_optimal) = gradient_descent(X, y, params, learning_rate, iterations)
	return params_optimal

X, y, _ = get_x_and_y(pcr_documents, pcr_summaries, pcr_oracles, 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
trained_params = train(X_train, y_train)
y_pred = [predict(x, trained_params) for x in X_test]
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
