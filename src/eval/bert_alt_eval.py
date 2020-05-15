from utils import *
from model import *
from trainbert import *
from testbert import *
from bertsystemdataset import *
import math
import argparse
import os
from collections import Counter
from itertools import accumulate, permutations

output_dir = './bert_model_quakes'
TYPES = [0, 1, 2, 3, 4, 5]

def predict(test_loader, typ):
	device = torch.device("cuda")

	print('[II] Start testing')
	
	model = BertForSequenceClassification.from_pretrained(output_dir + str(typ) + '/').cuda()
	model.eval()

	preds = []

	# Predict 
	for batch in test_loader:
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
		
		pred = np.argsort(logits, axis=0).flatten()[::-1]
		preds.append(pred)

	return preds


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--datadir', default='pcr_data')

	args = parser.parse_args()

	DATA_DIR = args.datadir

	print('Loading data...')
	documents, summaries, oracles, types = load(DATA_DIR)
	test_files = os.listdir(DATA_DIR + '/bert_processed/test/')
	test_indices = [int(file[:-4]) for file in test_files if file.endswith('.npy')]
	test_loader = create_datasets(DATA_DIR, test_indices)
	summaries = [summaries[i] for i in test_indices]
	documents = [documents[i] for i in test_indices]
	types = [types[i] for i in test_indices]
	oracles = [oracles[i] for i in test_indices]

	scores = []
	lens = []
	all_preds = {}
	for t in TYPES:
		all_preds[t] = predict(test_loader, typ=t)

	for i, t in enumerate(types):
		d = dict(Counter([x[0] for x in t]))
		nongarbage = [i for i in range(len(t)) if t[i][0] != 6]
		oracle = [oracles[i][j] for j in nongarbage]
		t_ = [t[j][0] for j in nongarbage]
		score = 0.0
		for j in range(len(nongarbage)):
			pred = all_preds[t_[j]][i]
			pos = list(pred).index(oracle[j]) + 1
			score += d[t_[j]]/len(documents[i])
			d[t_[j]] -= 1
			score -= pos/len(documents[i])
		scores.append(score/len(documents[i]))
		lens.append(len(nongarbage))
	print(np.mean(scores))
	print(np.mean(lens))

