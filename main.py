from utils import *
from model import *
from train import *
from test import *
from summarizationdataset import *
import math

data_dir = 'pcr_data/'
sent_type = -1
BATCH_SIZE = 16
EPOCHS = 100
FEATURES = None
MINI = False

if __name__ == '__main__':
	print('Loading data...')
	documents, summaries, oracles = load(data_dir)
	train_loader, test_loader, valid_loader, available_indices, indices_test = create_datasets(data_dir, oracles, sent_type, BATCH_SIZE, MINI)
	for inputs, mask, targets in train_loader:
		FEATURES = inputs[0].shape[1]
		break
	train(train_loader, valid_loader, num_features=FEATURES, n_epochs=EPOCHS, batch_size=BATCH_SIZE)
	test_scores = test(test_loader, num_features=FEATURES).cpu().detach().numpy()

	oracle_rouges = []
	model_rouges = []
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

