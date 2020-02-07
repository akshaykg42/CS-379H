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
	X, y, failed_indices = get_features_and_labels(documents, summaries, oracles, sent_type)
	X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, np.arange(len(X)))
	pickle.dump(X_train, open('X_train.pkl', 'wb'))
	pickle.dump(X_test, open('X_test.pkl', 'wb'))
	pickle.dump(y_train, open('y_train.pkl', 'wb'))
	pickle.dump(y_test, open('y_test.pkl', 'wb'))
	pickle.dump(indices_train, open('indices_train.pkl', 'wb'))
	pickle.dump(indices_test, open('indices_test.pkl', 'wb'))
	pickle.dump(failed_indices, open('failed_indices.pkl', 'wb'))	
	train(X_train, y_train)
	#X_test = pickle.load(open('X_test.pkl', 'rb'))
	#y_test = pickle.load(open('y_test.pkl', 'rb'))
	#indices_train = pickle.load(open('indices_train.pkl', 'rb'))
	#indices_test = pickle.load(open('indices_test.pkl', 'rb'))
	#failed_indices = pickle.load(open('failed_indices.pkl', 'rb'))
	test_scores = test(X_test, y_test).cpu().detach().numpy()
	
	documents = [documents[i] for i in range(len(documents)) if i not in failed_indices]
	summaries = [summaries[i] for i in range(len(summaries)) if i not in failed_indices]
	oracles = [oracles[i] for i in range(len(oracles)) if i not in failed_indices]

	for i, index in enumerate(indices_test):
		document = documents[index]
		summary = summaries[index]
		oracle_index = oracles[index][sent_type]
		model_scores = test_scores[i]
		document_sentences = tokenizer.tokenize(document)
		summary_sentences = tokenizer.tokenize(summary)
		ground_truth_sentence = summary_sentences[sent_type]
		print('Ground Truth Sentence: {}\n'.format(ground_truth_sentence))
		oracle_sentence = document_sentences[oracle_index]
		model_best_sentence_indices = model_scores.flatten().argsort()[-5:][::-1]
		model_best_scores = np.sort(model_scores.flatten())[-5:][::-1]
		model_best_probabilities = [math.exp(score) for score in model_best_scores]
		model_best_sentences = [document_sentences[j] for j in model_best_sentence_indices]
		oracle_rouge = get_rouge(oracle_sentence, ground_truth_sentence, '1', 'f')
		print('Oracle Sentence rouge({}):'.format(oracle_rouge))
		print(oracle_sentence)
		print()
		for model_sentence, model_probability in zip(model_best_sentences, model_best_probabilities):
			model_rouge = get_rouge(model_sentence, ground_truth_sentence, '1', 'f')
			print('Model Sentence p[{}] rouge({}):'.format(model_probability, model_rouge))
			print(model_sentence)
			print()
		if(oracle_sentence in model_best_sentences):
			print('ORACLE IN TOP 5')
		print('----------------------------------------------------------------------\n')

