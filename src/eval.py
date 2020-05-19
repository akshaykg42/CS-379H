import os
import argparse
from rouge import Rouge
from torch import nn, optim
import allennlp_models.coref
from models.linear import *
from collections import Counter
from models.data_loader import *
from transformers import BertForSequenceClassification
#from allennlp.predictors.predictor import Predictor
#predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz")

rouge = Rouge()
rouge_metric = 'f'
rouge_type = 'rouge-1'
coref_keywords = ['the', 'it', 'they', 'its', 'a']

# Linear prediction function, feeds each example in dataloader to linear model for a topic
# Returns, for each example document in the dataloader, a list of indices of its sentences
# sorted in decreasing order of the model's predicted relevance to the topic
def predict_linear(dataloader, topic):
	print('Running Linear model for topic {} on test data from the {} dataset...'.format(topic, dataloader.dataset.dataset_name))

	load_path = '../models/{}/linear/{}.th'.format(dataloader.dataset.dataset_name, topic)

	for inputs, mask, targets in dataloader:
		num_features = inputs[0].shape[1]
		break

	model = LinearModel(num_features).cuda()
	model.load_state_dict(torch.load(load_path))
	model.eval()

	preds = []
	for inputs, mask, targets in dataloader:
		scores, pred = model(inputs, mask)
		preds.append(np.argsort(scores.detach().cpu().numpy().flatten().tolist())[::-1])

	return preds

# BERT prediction function, feeds each example in dataloader to linear model for a topic
# Returns, for each example document in the dataloader, a list of indices of its sentences
# sorted in decreasing order of the model's predicted relevance to the topic
def predict_bert(dataloader, topic):
	print('Running BERT model for topic {} on test data from the {} dataset...'.format(topic, dataloader.dataset.dataset_name))
	
	device = torch.device("cuda")

	load_path = '../models/{}/bert/{}/'.format(dataloader.dataset.dataset_name, topic)
	
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

# Gets index of an array in a subarray
def get_index(array, subarray):
	for i in range(len(array) - len(subarray) + 1):
		if(array[x:x+len(subarray)] == subarray):
			return i

# Get context for sentence #index in document
# TODO: Better interface for keywords?
def get_context(index, document):
	doc = ' '.join(document)
	out = set()
	# Coref Resolution for each sentence so we can get sentence lengths
	sent_corefs = []
	for sent in document:
		sent_corefs.append(predictor.predict(sent))
	sent_lens = [0]
	# Get length of each sentence tokenized by allennlp and keep a running sum in sent_lens
	for i in range(len(sent_corefs)):
		sent_coref = sent_corefs[i]
		sent_lens.append(sent_lens[i] + len(sent_coref['document']))
	sent = document[index]
	sent_coref = predictor.predict(sent)
	doc_coref = predictor.predict(doc)
	# Get index of tokenized sentence in tokenized document 
	start = get_index(doc_coref['document'], sent_coref['document'])
	end = start + len(sent_coref['document'])
	sent_clusters = []
	# Get all coref clusters in document
	for cluster in doc_coref['clusters']:
		# Check every span (reference to an entity) in every cluster
		for span in cluster:
			# If it is inside the sentence we're looking at, pick up this cluster
			if(span[0] >= start and span[1] <= end):
				sent_clusters.append(cluster)
				break
	for cluster in list(sent_clusters):
		# For each span in each cluster, see what sentence it is in, if it occurs in/after our sentence, stop
		for span in cluster:
			idx = 0
			while(idx < len(sent_lens) - 1 and sent_lens[idx+1] <= span[0]):
				idx += 1
			if(idx < index):
				# Get the first word of the span, if it's in our keywords (hardcoded) then keep the sentence
				firstword = doc_coref['document'][span[0]].lower()
				if(firstword in coref_keywords):
					out.add(idx)
			else:
				break
	return out

def get_rouge(hypothesis, reference, rougetype, scoretype):
	return rouge.get_scores(hypothesis, reference)[0][rougetype][scoretype]

'''
Constructs a summary by getting the best sentence for each topic in topics, adding context
using coref resolution if selected, and concatenating the sentences after sorting them
Evaluation is done by calculating ROUGE-1, ROUGE-2 and ROUGE-L F1 scores with respect to gold summaries
'''
# TODO: More configurability for context
def vanilla_eval(dataloader, predict_fn, topics, documents, summaries, context=False):
	best_sentence_per_topic = []
	# Get best sentence per topic for each document for each topic
	for topic in topics:
		best_sentence_per_topic.append([prediction[0] for prediction in predict_fn(dataloader, topic)])

	# Indices that make up each generated summary (without context)
	summary_indices = [[best_sentence_per_topic[i][j] for i in range(len(topics))] for j in range(len(documents))]

	if(context):
		indices_with_context = []
		for i in range(len(documents)):
			all_indices = set(summary_indices[i])
			# For each sentence selected by a model, get context and keep the first 2 sentences
			for k in summary_indices[i]:
				context = sorted(list(get_context(k, documents[i])))[:2]
				all_indices = all_indices.union(context)
			indices_with_context.append(all_indices)
		summary_indices = [list(item) for item in indices_with_context]
	
	# Sort summary indices
	summary_indices = [sorted(indices) for indices in summary_indices]

	# Construct summaries as strings
	model_summaries = [' '.join([documents[i][k] for k in summary_indices[i]]) for i in range(len(documents))]

	# Calculate ROUGE directly wrt gold summaries
	rouge1 = np.mean([get_rouge(model_summaries[i], ' '.join(summaries[i]), 'rouge-1', 'f') for i in range(len(documents))])
	rouge2 = np.mean([get_rouge(model_summaries[i], ' '.join(summaries[i]), 'rouge-2', 'f') for i in range(len(documents))])
	rougel = np.mean([get_rouge(model_summaries[i], ' '.join(summaries[i]), 'rouge-l', 'f') for i in range(len(documents))])

	print('ROUGE-1 F1: {}'.format(rouge1*100))
	print('ROUGE-2 F1: {}'.format(rouge2*100))
	print('ROUGE-L F1: {}'.format(rougel*100))

	return model_summaries

# Try all permutations for summary indices to see which maximizes pairwise ROUGE with summary sentences
def optimize_pred(document, summary, pred):
	try:
		if(len(pred) <= 9):
			options = list(permutations(pred))
			best_option = options[0]
			best_score = sum([get_rouge(summary[j], document[options[0][j]], rouge_type, rouge_metric) for j in range(len(summary))])
			for option in options:
				score = sum([get_rouge(summary[j], document[option[j]], rouge_type, rouge_metric) for j in range(len(summary))])
				if(score > best_score):
					best_score = score
					best_option = option
			return best_option
	except ValueError:
		pass
	return pred

'''
Constructs a summary by attempting to reconstruct the gold summary using the topic representation -
for each distinct topic in the topic representation, we get the top k sentences for that topic from the
document where k is the number of times the topic occurs in the representation. We then optimize the 
indices using a similar procedure as in our oracle construction and return the result
Evaluation is done by calculating ROUGE-1, ROUGE-2 and ROUGE-L F1 scores with respect to gold summaries
'''
def reconstruct_eval(dataloader, predict_fn, topics, documents, summaries, topic_representations):
	per_topic_rankings = {}
	# Get best sentence per topic for each document for each topic
	for topic in topics:
		per_topic_rankings[topic] = predict_fn(dataloader, topic)

	rouge1 = []
	rouge2 = []
	rougel = []

	model_summaries = []

	for index, representation in enumerate(topic_representations):
		# Get frequency of each topic in the summary's topic representation
		topic_freq_in_summary = dict(Counter(sum([topic for topic in representation], [])))
		summary_indices = []
		for topic in topic_freq_in_summary:
			# If a topic isn't in the ones we care about, move on
			if(topic not in topics):
				continue
			# Get the top k sentences from the predictions for this document for this topic
			rankings = per_topic_rankings[topic][index][:topic_freq_in_summary[topic]]
			summary_indices.extend(rankings)
		
		# Remove sentences from the summary that aren't of the topics we care about - this is for evaluation
		summary_shortened = [sentence for j, sentence in enumerate(summaries[index]) if set(topics) & set(representation[j])]
		
		# If no sentences we care about, skip
		if(len(summary_shortened) == 0):
			continue

		# Optimize order of indices to maximize pairwise ROUGE-1 F1 with gold summary sentences
		optimized_indices = optimize_pred(documents[index], summary_shortened, summary_indices)
		model_summary = ' '.join([documents[index][j] for j in optimized_indices])
		model_summaries.append(model_summary)

		rouge1.append(get_rouge(' '.join(summary_shortened), model_summary, 'rouge-1', 'f'))
		rouge2.append(get_rouge(' '.join(summary_shortened), model_summary, 'rouge-2', 'f'))
		rougel.append(get_rouge(' '.join(summary_shortened), model_summary, 'rouge-l', 'f'))

	# Calculate ROUGE
	rouge1 = (np.mean(rouge1))
	rouge2 = (np.mean(rouge2))
	rougel = (np.mean(rougel))

	print('ROUGE-1 F1: {}'.format(rouge1*100))
	print('ROUGE-2 F1: {}'.format(rouge2*100))
	print('ROUGE-L F1: {}'.format(rougel*100))

	return model_summaries

'''
Constructs a summary by attempting to reconstruct the gold summary using the topic representation -
for each distinct topic in the topic representation, we get the top k sentences for that topic from the
document where k is the number of times the topic occurs in the representation. We then optimize the 
indices using a similar procedure as in our oracle construction and return the result
Evaluation is done by summing the following over all topics that occur in the summary that we care about, and then dividing by the number of sentences (normalizing for length):
If there were k sentences of the topic in the summary, and the ranks of the oracle sentence for each as predicted by
the model are [r_1, ..., r_k] and the number of sentences in the document is N then the score for that topic is
(1/n - r_1/n) + (2/n - r_2/n) + ... + (k/n - r_k/n) i.e. penalized for rank being too far from the top
notice that order of r_i doesn't matter, also score <= 0 with less negative score being better
'''
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
		# Get frequency of each topic in the summary's topic representation
		topic_freq_in_summary = dict(Counter(sum([topic for topic in representation], [])))

		summary_indices = []
		
		for topic in topic_freq_in_summary:
			# If a topic isn't in the ones we care about, move on
			if(topic not in topics):
				continue
			# Get the top k sentences from the predictions for this document for this topic
			rankings = per_topic_rankings[topic][i][:topic_freq_in_summary[topic]]
			summary_indices.extend(rankings)
		
		# Get indices of sentences that are of topics we care about - this is for evaluation
		nongarbage = [j for j in range(len(representation)) if set(topics) & set(representation[j])]
		
		summary_shortened = [summaries[i][j] for j in nongarbage]
		
		# If no sentences we care about, skip
		if(len(summary_shortened) == 0):
			continue

		# Optimize order of indices to maximize pairwise ROUGE-1 F1 with gold summary sentences
		optimized_indices = optimize_pred(documents[i], summary_shortened, summary_indices)
		model_summary = ' '.join([documents[i][j] for j in optimized_indices])
		model_summaries.append(model_summary)

		# Score calculation explained above function
		score = 0.0
		for j in nongarbage:
			topic = list(set(representation[j]) & set(topics))[0]
			ranking = per_topic_rankings[topic][i]
			oracle_pos = list(ranking).index(oracles[i][j]) + 1
			score += topic_freq_in_summary[topic]/len(documents[i])
			topic_freq_in_summary[topic] -= 1
			score -= oracle_pos/len(documents[i])
		scores.append(score/len(documents[i]))

	print('SCORE: {}'.format(np.mean(scores)))

	return model_summaries

# Get all topics for which a model has been trained for a given dataset and model type
def get_all_topics(dataset_name, model_type):
	model_files = os.listdir('../models/{}/{}/'.format(dataset_name, model_type))
	if(model_type == 'linear'):
		return sorted([int(file[:-3]) for file in model_files if file != 'checkpoint.pt'])
	elif(model_type == 'bert'):
		return sorted([int(file) for file in model_files])

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-dataset_name')
	parser.add_argument('-model_type')
	parser.add_argument('-mode')
	# Topics to summarize - only used for vanilla evaluation
	# If left empty for vanilla then all topics are used
	parser.add_argument('-topics', nargs='*', type=int)
	# Whether or not to write the summaries generated to a file in ../results/{dataset_name}/
	parser.add_argument('-write', action='store_true', default=False)

	args = parser.parse_args()

	dataset_name, model_type, mode, topics, write = \
		args.dataset_name, args.model_type, args.mode, args.topics, args.write

	# Topics only used for vanilla eval, if not entered for vanilla i.e. is None then use all as well
	if(mode != 'vanilla' or topics is None):
		topics = get_all_topics(dataset_name, model_type)

	# Setting topic=None makes the loader return all test examples, not just the ones from a specific topic
	# Batch size hardcoded to 1 because it's easier to process output that way, and we don't have that many examples
	dataloader = create_loader(dataset_name, model_type, 'test', topic=None, batch_size=1)

	# Where to write results if we do
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

	# Filter documents/summaries/oracles/topic representations to only those from the test set
	documents = [documents[k] for k in dataloader.dataset.labels]
	summaries = [summaries[k] for k in dataloader.dataset.labels]
	oracles = [oracles[k] for k in dataloader.dataset.labels]
	topic_representations = [topic_representations[k] for k in dataloader.dataset.labels]

	# Prediction function based on model type
	if(model_type == 'linear'):
		predict_fn = predict_linear
	elif(model_type == 'bert'):
		predict_fn = predict_bert

	if(mode == 'vanilla'):
		model_summaries = vanilla_eval(dataloader, predict_fn, topics, documents, summaries)
	elif(mode == 'reconstruct'):
		model_summaries = reconstruct_eval(dataloader, predict_fn, topics, documents, summaries, topic_representations)
	elif(mode == 'ranking'):
		model_summaries = ranking_eval(dataloader, predict_fn, topics, documents, summaries, oracles, topic_representations)

	# Write summaries to a file, specifying model type, mode and topics
	if(write):
		save_path = '{}/{}_{}_{}.txt'.format(save_dir, model_type, mode, str(topics))
		with open(save_path, 'w') as outfile:
			for model_summary in model_summaries:
				outfile.write(model_summary)
				outfile.write('\n\n')

	






