from utils import *
from model import *
from systemdataset import *
import math
import argparse
import os
from collections import Counter
from itertools import accumulate, permutations

dirname = os.path.dirname(os.path.abspath(__file__))
model_name = 'linear_model_quakes'
TYPES = [0, 1, 2, 3, 4, 5]

def predict(test_loader, num_features, typ):
	print('[II] Start testing')
	
	model = OracleSelectorModel(num_features).cuda()
	model.load_state_dict(torch.load(os.path.join(dirname, model_name + str(typ) + '.th')))
	model.eval()

	preds = []
	for inputs, mask in test_loader:
		scores, pred = model(inputs, mask)
		preds.append(np.argsort(scores.detach().cpu().numpy().flatten().tolist()))
	return preds

def optimize_pred(document, summary, pred):
	try:
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
	except ValueError:
		pass
	return pred


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
	for inputs, mask in test_loader:
		FEATURES = inputs[0].shape[1]
		break

	rouge1_avg = 0.0
	rouge2_avg = 0.0
	rougel_avg = 0.0

	rouge1 = []
	rouge2 = []
	rougel = []
	all_preds = []
	for t in TYPES:
		all_preds.append(predict(test_loader, num_features=FEATURES, typ=t))

	for i, t in enumerate(types):
		d = dict(Counter([x[0] for x in t]))
		out = []
		for k in d:
			if(k == 6):
				continue
			preds = all_preds[k][i][::-1][:d[k]]
			out.extend(preds)
		s_ = [x for j, x in enumerate(summaries[i]) if t[j][0] != 6]
		if(len(s_) == 0):
			continue
		optimized_pred = optimize_pred(documents[i], s_, out)
		rouge1.append(rouge.get_scores(' '.join(s_), ' '.join([documents[i][p] for p in optimized_pred]))[0]['rouge-1']['f'])
		rouge2.append(rouge.get_scores(' '.join(s_), ' '.join([documents[i][p] for p in optimized_pred]))[0]['rouge-2']['f'])
		rougel.append(rouge.get_scores(' '.join(s_), ' '.join([documents[i][p] for p in optimized_pred]))[0]['rouge-l']['f'])

	rouge1_avg += (np.mean(rouge1))
	rouge2_avg += (np.mean(rouge2))
	rougel_avg += (np.mean(rougel))

	print(rouge1_avg)
	print(rouge2_avg)
	print(rougel_avg)

