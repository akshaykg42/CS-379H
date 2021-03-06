from utils import *
from model import *
from systemdataset import *
import math
import argparse
import os
from allennlp.predictors.predictor import Predictor

dirname = os.path.dirname(os.path.abspath(__file__))
predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz")
predictor._dataset_reader._token_indexers['token_characters']._min_padding_length = 10
model_name = 'linear_model_quakes'
TYPES = [2, 5]
#coref_keywords = ['the', 'it', 'they', 'its', 'a']
coref_keywords = ['the', 'it', 'they', 'its', 'a', 'he', 'she', 'they']
def predict(test_loader, num_features, typ):
	print('[II] Start testing')
	
	model = OracleSelectorModel(num_features).cuda()
	model.load_state_dict(torch.load(os.path.join(dirname, model_name + str(typ) + '.th')))
	model.eval()

	preds = []
	for inputs, mask in test_loader:
		scores, pred = model(inputs, mask)
		preds.append(pred.detach().cpu().numpy().flatten().tolist()[0])

	return preds

def get_index(array, subarray):
	return [x for x in range(len(array) - len(subarray) + 1) if array[x:x+len(subarray)] == subarray][0]


def get_context(sentencenumber, document):
	doc = ' '.join(document)
	out = set()
	sent_corefs = []
	for sent in document:
		sent_corefs.append(predictor.predict(sent))
	sent_lens = [0]
	for i in range(len(sent_corefs)):
		sent_coref = sent_corefs[i]
		sent_lens.append(sent_lens[i] + len(sent_coref['document']))
	sent = document[sentencenumber]
	sent_coref = predictor.predict(sent)
	doc_coref = predictor.predict(doc)
	start = get_index(doc_coref['document'], sent_coref['document'])
	end = start + len(sent_coref['document'])
	sent_clusters = []
	for cluster in doc_coref['clusters']:
		for span in cluster:
			if(span[0] >= start and span[1] <= end):
				sent_clusters.append(cluster)
				break
	for cluster in list(sent_clusters):
		for span in cluster:
			idx = 0
			while(idx < len(sent_lens) - 1 and sent_lens[idx+1] <= span[0]):
				idx += 1
			if(idx < sentencenumber):
				firstword = doc_coref['document'][span[0]].lower()
				if(firstword in coref_keywords):
					out.add(idx)
			else:
				break
	return out


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--datadir', default='pcr_data')

	args = parser.parse_args()

	DATA_DIR = args.datadir

	print('Loading data...')
	documents, summaries, oracles, types = load(DATA_DIR)
	test_files = os.listdir(DATA_DIR + '/processed/test/')
	test_indices = [int(file[:-4]) for file in test_files if file.endswith('.npy')]
	test_loader = create_datasets(DATA_DIR, test_indices)
	summaries = [summaries[i] for i in test_indices]
	for inputs, mask in test_loader:
		FEATURES = inputs[0].shape[1]
		break

	rouge1_avg = 0.0
	rouge2_avg = 0.0
	rougel_avg = 0.0

	all_preds = []
	for t in TYPES:
		all_preds.append(predict(test_loader, num_features=FEATURES, typ=t))

	#model_summaries = [[documents[j][all_preds[i][j]] for i in range(len(TYPES))] for j in range(len(indices))]

	model_summary_indices = [[all_preds[i][j] for i in range(len(TYPES))] for j in range(len(test_indices))]
	contexts = []
	for i, j in enumerate(test_indices):
		contexts.append(model_summary_indices[i])
		# out = set(model_summary_indices[i])
		# for k in model_summary_indices[i]:
		# 	context = sorted(list(get_context(k, documents[j])))[:2]
		# 	out = out.union(context)
		# contexts.append(out)

	contexts = [sorted(list(context)) for context in contexts]

	model_summaries = [' '.join([documents[j][k] for k in contexts[i]]) for i, j in enumerate(test_indices)]

	# for i, j in enumerate(test_indices):
	# 	print('\n'.join([documents[j][k] for k in contexts[i]]))
	# 	print('\n--------------------------------------------------\n')

	rouge1 = [rouge.get_scores(model_summaries[i], ' '.join(summaries[i]))[0]['rouge-1']['f'] for i in range(len(test_indices))]
	rouge2 = [rouge.get_scores(model_summaries[i], ' '.join(summaries[i]))[0]['rouge-2']['f'] for i in range(len(test_indices))]
	rougel = [rouge.get_scores(model_summaries[i], ' '.join(summaries[i]))[0]['rouge-l']['f'] for i in range(len(test_indices))]

	rouge1_avg += (np.mean(rouge1))
	rouge2_avg += (np.mean(rouge2))
	rougel_avg += (np.mean(rougel))
	print(rouge1_avg)
	print(rouge2_avg)
	print(rougel_avg)
	'''
	for m_s in model_summaries:
		print('\n'.join(m_s))
		print('\n--------------------------------------------------\n')
	'''
	'''
	oracle_rouges = []
	model_rouges = []

	labels = []
	ground_truths = []
	for i, (t, o) in enumerate(list(zip(types, oracles))):
		for j, t_ in enumerate(t):
			if(TYPE in t_):
				ground_truths.append(j)
				labels.append(o[j])
	
	for i, index in enumerate(indices_test):
		document = documents[index]
		summary = summaries[index]
		oracle_index = labels[index]
		model_scores = test_scores[i]
		document_sentences = tokenizer.tokenize(document)
		summary_sentences = tokenizer.tokenize(summary)
		ground_truth_sentence = summary_sentences[ground_truths[index]]
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

