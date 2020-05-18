import os
import argparse
from torch import nn, optim
from models.linear import *
from models.data_loader import *
from transformers import BertForSequenceClassification
from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz")
predictor._dataset_reader._token_indexers['token_characters']._min_padding_length = 10

def predict_linear(dataloader, topic):
	load_path = '../models/{}/linear/{}.th'.format(dataloader.dataset_name, topic)

	for inputs, mask, targets in dataloader:
		num_features = inputs[0].shape[1]
		break

	model = LinearModel(num_features).cuda()
	model.load_state_dict(torch.load(load_path))
	model.eval()

	preds = []
	for inputs, mask in test_loader:
		scores, pred = model(inputs, mask)
		preds.append(np.argsort(scores.detach().cpu().numpy().flatten().tolist())[::-1])

	return preds

def predict_bert(dataloader, topic):
	device = torch.device("cuda")

	load_path = '../models/{}/bert/{}/'.format(dataloader.dataset_name, topic)
	
	model = BertForSequenceClassification.from_pretrained(load_path).cuda()
	model.eval()

	preds = []

	# Predict 
	for batch in dataloader:
		# Add batch to GPU
		b_input_ids = batch[0].to(device)
		b_input_mask = batch[1].to(device)
		
		# Telling the model not to compute or store gradients, saving memory and 
		# speeding up prediction
		with torch.no_grad():
			# Forward pass, calculate logit predictions
			outputs = model(b_input_ids, token_type_ids=None, 
							attention_mask=b_input_mask)
		
		logits = outputs[0]
		
		# Move logits and labels to CPU
		logits = logits.detach().cpu().numpy()
		
		pred = list(np.argsort(logits, axis=0).flatten()[::-1])
		preds.append(pred)

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

def vanilla_eval(dataloader, predict_fn, topics, documents, summaries):
	best_sentence_per_topic = []
	for topic in topics:
		best_sentence_per_topic.append([prediction[0] for prediction in predict_fn(dataloader, topic)])

	summary_indices = [[best_sentence_per_topic[i][j] for i in range(len(topics))] for j in range(len(documents))]

	indices_with_context = []
	for i in len(documents):
		all_indices = set(summary_indices[i])
		for k in summary_indices[i]:
			context = sorted(list(get_context(k, documents[i])))[:2]
			all_indices = all_indices.union(context)
		indices_with_context.append(all_indices)

	indices_with_context = [sorted(list(indices)) for indices in indices_with_context]

	model_summaries = [' '.join([documents[i][k] for k in indices_with_context[i]]) for i in len(documents)]

	rouge1 = np.mean([rouge.get_scores(model_summaries[i], ' '.join(summaries[i]))[0]['rouge-1']['f'] for i in range(len(documents))])
	rouge2 = np.mean([rouge.get_scores(model_summaries[i], ' '.join(summaries[i]))[0]['rouge-2']['f'] for i in range(len(documents))])
	rougel = np.mean([rouge.get_scores(model_summaries[i], ' '.join(summaries[i]))[0]['rouge-l']['f'] for i in range(len(documents))])

	print('ROUGE-1 F1: {}'.format(rouge1*100))
	print('ROUGE-2 F1: {}'.format(rouge2*100))
	print('ROUGE-L F1: {}'.format(rougel*100))

	return model_summaries

def optimize_summary(document, summary, pred):
	if(len(pred) <= 9):
		options = list(permutations(pred))
		best_option = options[0]
		best_score = sum([rouge.get_scores(summary[j], document[options[0][j]])[0]['rouge-1']['f'] for j in range(len(summary))])
		for option in options:
			score = sum([rouge.get_scores(summary[j], document[option[j]])[0]['rouge-1']['f'] for j in range(len(summary))])
			if(score > best_score):
				best_score = score
				best_option = option
		return best_option
	return pred

def reconstruct_eval(dataloader, predict_fn, topics, documents, summaries, topic_representations):
	per_topic_rankings = {}
	for topic in topics:
		per_topic_rankings[topic] = predict_fn(dataloader, topic)

	rouge1 = []
	rouge2 = []
	rougel = []

	model_summaries = []

	for index, representation in enumerate(topic_representations):
		topic_freq_in_summary = dict(Counter(sum([[topic for topic in representation]], [])))
		summary_indices = []
		for topic in topic_freq_in_summary:
			if(topic not in topics):
				continue
			rankings = per_topic_rankings[topic][index][:topic_freq_in_summary[topic]]
			summary_indices.extend(rankings)
		
		summary_shortened = [sentence for j, sentence in enumerate(summaries[index]) if set(topics).insersection(set(representation[j]))]
		
		if(len(summary_shortened) == 0):
			continue

		optimized_indices = optimize_pred(documents[index], summary_shortened, summary_indices)
		model_summary = ' '.join([documents[index][j] for j in optimized_indices])
		model_summaries.append(model_summary)

		rouge1.append(rouge.get_scores(' '.join(s_), model_summary)[0]['rouge-1']['f'])
		rouge2.append(rouge.get_scores(' '.join(s_), model_summary)[0]['rouge-2']['f'])
		rougel.append(rouge.get_scores(' '.join(s_), model_summary)[0]['rouge-l']['f'])

	rouge1_avg = (np.mean(rouge1))
	rouge2_avg = (np.mean(rouge2))
	rougel_avg = (np.mean(rougel))

	print('ROUGE-1 F1: {}'.format(rouge1*100))
	print('ROUGE-2 F1: {}'.format(rouge2*100))
	print('ROUGE-L F1: {}'.format(rougel*100))

	return model_summaries

def ranking_eval(dataloader, predict_fn, topics, documents, summaries, oracles, topic_representations):
	per_topic_rankings = {}
	for topic in topics:
		per_topic_rankings[topic] = predict_fn(dataloader, topic)

	rouge1 = []
	rouge2 = []
	rougel = []

	model_summaries = []

	scores = []

	for i, representation in enumerate(topic_representations):
		topic_freq_in_summary = dict(Counter(sum([[topic for topic in representation]], [])))

		summary_indices = []
		
		for topic in topic_freq_in_summary:
			if(topic not in topics):
				continue
			rankings = per_topic_rankings[topic][i][:topic_freq_in_summary[topic]]
			summary_indices.extend(rankings)
		
		nongarbage = [j for j in range(len(representation)) if set(topics).insersection(set(representation[j]))]
		summary_shortened = [summaries[i][j] for j in nongarbage]
		
		if(len(summary_shortened) == 0):
			continue

		optimized_indices = optimize_pred(documents[i], summary_shortened, summary_indices)
		model_summary = ' '.join([documents[i][j] for j in optimized_indices])
		model_summaries.append(model_summary)

		score = 0.0
		for j in nongarbage:
			topic = list(set(representation[j]).intersection(set(topics)))[0]
			ranking = per_topic_rankings[topic][i]
			oracle_pos = list(ranking).index(oracles[i][j]) + 1
			score += topic_freq_in_summary[topic]/len(documents[i])
			topic_freq_in_summary[topic] -= 1
			score -= oracle_pos/len(documents[i])
		scores.append(score/len(documents[i]))

	print('SCORE: {}'.format(np.mean(scores)))

	return model_summaries

def get_all_topics(dataset_name, model_type):
	model_files = os.listdir('../models/{}/{}/'.format(dataset_name, model_type))
	if(model_type == 'linear'):
		return [int(file[:-3]) for file in model_files]
	elif(model_type == 'bert'):
		return [int(file) for file in model_files]

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-dataset_name')
	parser.add_argument('-model_type')
	parser.add_argument('-mode')
	parser.add_argument('-topics', narge='*', type=int)
	parser.add_argument('write', action='store_true', default=False)

	args = parser.parse_args()

	dataset_name, model_type, mode, topics, write = \
		args.dataset_name, args.model_type, args.mode, args.topics, args.write

	if(topics == None):
		topics = get_all_topics(dataset_name, model_type)

	dataloader = create_loader(dataset_name, model_type, 'test', topic, batch_size=1)

	save_dir = '../results/{}'.format(dataset_name)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	json_path = '../data/{}/raw/'.format(dataset_name)
	
	with open(json_path + 'documents.json') as json_file:
		documents = json.load(json_file)
	with open(json_path + 'summaries.json') as json_file:
		summaries = json.load(json_file)
	with open(json_path + 'oracles.json') as json_file:
		oracles = json.load(json_file)
	with open(json_path + 'topics.json') as json_file:
		topic_representations = json.load(json_file)

	documents = [documents[k] for k in dataloader.labels]
	summaries = [summaries[k] for k in dataloader.labels]
	oracles = [oracles[k] for k in dataloader.labels]
	topic_representations = [topic_representations[k] for k in dataloader.labels]

	if(model_type == 'linear'):
		predict_fn = predict_linear
	elif(model_type == 'bert'):
		predict_fn = predict_bert

	if(mode == 'vanilla'):
		model_summaries = vanilla_eval(dataloader, predict_fn, topics, documents, summaries)
	elif(mode == 'reconstruct'):
		model_summaries = reconstruct_eval(dataloader, predict_fn, topics, documents, summaries, topic_representations)
	elif(mode == 'ranking'):
		model_summaries = ranking_eval(dataloader, predict_fn, topics, documents, summaries, oracles, topic_representations):

	if(write):
		save_path = '{}/{}_{}_{}.txt'.format(save_dir, model_type, mode, str(topics))
		with open(save_path, 'w') as outfile:
			for model_summary in model_summaries:
				outfile.write(model_summary)
				outfile.write('\n\n')

	






