from utils import *
from model import *
from train import *
from test import *
import math
from sklearn.model_selection import train_test_split

sent_type = 0

if __name__ == '__main__':
	print('Loading data...')
	documents, summaries, oracles = load('pcr_documents.pkl', 'pcr_summaries.pkl', 'pcr_oracles.pkl')
	X, y = get_features_and_labels(documents, summaries, oracles, sent_type)
	X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, np.arange(len(X)))
	train(X_train, y_train)
	test_scores = test(X_test, y_test).detach().numpy()

	for i, index in enumerate(indices_test):
		document = documents[index]
		summary = summaries[index]
		oracle_index = oracles[index][sent_type]
		model_scores = test_scores[i]
		document_sentences = tokenizer.tokenize(document)
		summary_sentences = tokenizer.tokenize(summary)
		ground_truth_sentence = summary[sent_type]
		oracle_sentence = document_sentences[oracle_index]
		model_best_sentence_indices = model_scores.flatten().argsort()[-5:][::-1]
		model_best_scores = np.sort(model_scores.flatten())[-5:][::-1]
		model_best_probabilities = [math.exp(score) for score in model_best_scores]
		model_best_sentences = [document_sentences[j] for j in model_best_sentences]
		oracle_rouge = get_rouge(oracle_sentence, ground_truth_sentence, '2', 'f')
		print('Oracle Sentence rouge({}):'.format(oracle_rouge))
		print(oracle_sentence)
		for model_sentence, model_probability in zip(model_best_sentences, model_best_probabilities):
			model_rouge = get_rouge(model_sentence, ground_truth_sentence, '2', 'f')
			print('Model Sentence p[{}] rouge({}):'.format(model_probability, model_rouge))
			print(model_sentence)
		print('----------------------------------------------------------------------')

