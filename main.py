from utils import *
from model import *
from train import *
from test import *
from summarizationdataset import *
import math
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', '--type', type=int, default=0)
	parser.add_argument('-b', '--batchsize', type=int, default=16)
	parser.add_argument('-e', '--epochs', type=int, default=100)
	parser.add_argument('-f', '--features', type=int, default=None)
	parser.add_argument('-d', '--datadir', default='pcr_data')
	parser.add_argument('-m', '--mini', action='store_true', default=False)

	args = parser.parse_args()

	DATA_DIR, TYPE, BATCH_SIZE, EPOCHS, FEATURES, MINI = \
		args.datadir, args.type, args.batchsize, args.epochs, args.features, args.mini

	print('Loading data...')
	documents, summaries, oracles, types = load(DATA_DIR)
	train_loader, test_loader, valid_loader, available_indices, indices_test = create_datasets(DATA_DIR, oracles, types, TYPE, BATCH_SIZE, MINI)
	for inputs, mask, targets in train_loader:
		FEATURES = inputs[0].shape[1]
		break
	train(train_loader, valid_loader, num_features=FEATURES, n_epochs=EPOCHS, batch_size=BATCH_SIZE)
	test_scores = [logits.cpu().detach().numpy() for logits in test(test_loader, num_features=FEATURES)]

	oracle_rouges = []
	model_rouges = []
	'''
	for i, index in enumerate(indices_test):
		document = documents[index]
		summary = summaries[index]
		oracle_index = oracles[index][TYPE]
		model_scores = test_scores[i]
		document_sentences = tokenizer.tokenize(document)
		summary_sentences = tokenizer.tokenize(summary)
		ground_truth_sentence = summary_sentences[TYPE]
		print('Ground Truth Sentence: {}\n'.format(ground_truth_sentence))
		oracle_sentence = document_sentences[oracle_index]
		model_best_sentence_indices = model_scores.flatten().argsort()[-5:][::-1]
		model_best_scores = np.sort(model_scores.flatten())[-5:][::-1]
		model_best_probabilities = [math.exp(score) for score in model_best_scores]
		model_best_sentences = [document_sentences[j] for j in model_best_sentence_indices]
		oracle_rouge = get_rouge(oracle_sentence, ground_truth_sentence, '1', 'f')
		oracle_rouges.append(oracle_rouge)
		print('Oracle Sentence rouge({}):'.format(oracle_rouge))
		print(oracle_sentence)
		print()
		tmp = []
		for model_sentence, model_probability in zip(model_best_sentences, model_best_probabilities):
			model_rouge = get_rouge(model_sentence, ground_truth_sentence, '1', 'f')
			tmp.append(model_rouge)
			print('Model Sentence p[{}] rouge({}):'.format(model_probability, model_rouge))
			print(model_sentence)
			print()
		model_rouges.append(tmp)
		if(oracle_sentence in model_best_sentences):
			print('ORACLE IN TOP 5')
		print('----------------------------------------------------------------------\n')
	avg_model_rouges = [np.mean(i) for i in model_rouges]
	min_model_rouges = [min(i) for i in model_rouges]
	max_model_rouges = [max(i) for i in model_rouges]
	print(oracle_rouges)
	print(avg_model_rouges)
	print(min_model_rouges)
	print(max_model_rouges)
	'''

