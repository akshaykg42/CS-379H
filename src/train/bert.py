from utils import *
from trainbert import *
from testbert import *
from bertdataset import *
import math
import argparse
from collections import Counter

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', '--type', type=int, default=0)
	parser.add_argument('-b', '--batchsize', type=int, default=1)
	parser.add_argument('-e', '--epochs', type=int, default=4)
	parser.add_argument('-d', '--datadir', default='pcr_data')
	parser.add_argument('-m', '--mini', action='store_true', default=False)

	args = parser.parse_args()

	DATA_DIR, TYPE, BATCH_SIZE, EPOCHS, MINI = \
		args.datadir, args.type, args.batchsize, args.epochs, args.mini

	print('Loading data...')
	documents, summaries, oracles, types = load(DATA_DIR)
	train_loader, test_loader, valid_loader, available_indices, indices_test, labels_test, ground_truths = create_datasets(DATA_DIR, oracles, types, TYPE, BATCH_SIZE, mini=MINI)
	train(train_loader, valid_loader, n_epochs=EPOCHS, batch_size=BATCH_SIZE, typ=TYPE)
	test_scores = test(test_loader, typ=TYPE)
	print(indices_test)

	oracle_rouges = []
	model_rouges = []

	# labels = []
	# ground_truths = []
	# c = Counter()
	# for i in indices_test:
	# 	if(i in s):
	# 		continue
	# 	else:
	# 		s.add(i)
	# 	t = types[i]
	# 	o = oracles[i]
	# 	for j, t_ in enumerate(t):
	# 		if(TYPE in t_):
	# 			ground_truths.append(j)
	# 			labels.append(o[j])
	# for i, (t, o) in enumerate(list(zip(types, oracles))):
	# 	if(i in available_indices):
	# 		for j, t_ in enumerate(t):
	# 			if(TYPE in t_):
	# 				ground_truths.append(j)
	# 				labels.append(o[j])
	
	for i, index in enumerate(indices_test):
		document_sentences = documents[index]
		summary_sentences = summaries[index]
		oracle_index = labels_test[i]
		model_scores = test_scores[i]
		ground_truth_sentence = summary_sentences[ground_truths[i]]
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
	
