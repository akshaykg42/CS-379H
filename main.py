from utils import *
from model import *
from train import *
from test import *
from summarizationdataset import *
import math

sent_type = 0
BATCH_SIZE = 16

if __name__ == '__main__':
	print('Loading data...')
	documents, summaries, oracles = load()
	train_loader, test_loader, valid_loader, available_indices = create_datasets(oracles, sent_type, BATCH_SIZE)
	train(train_loader, valid_loader)
	test_scores = test(test_loader).cpu().detach().numpy()
	
	documents = [documents[i] for i in range(len(documents)) if i in available_indices]
	summaries = [summaries[i] for i in range(len(summaries)) if i in available_indices]
	oracles = [oracles[i] for i in range(len(oracles)) if i in available_indices]

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

