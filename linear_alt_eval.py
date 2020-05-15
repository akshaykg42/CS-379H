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
		preds.append(np.argsort(scores.detach().cpu().numpy().flatten().tolist())[::-1])
	return preds


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
	documents = [documents[i] for i in test_indices]
	types = [types[i] for i in test_indices]
	oracles = [oracles[i] for i in test_indices]
	for inputs, mask in test_loader:
		FEATURES = inputs[0].shape[1]
		break

	scores = []
	lens = []
	all_preds = {}
	for t in TYPES:
		all_preds[t] = predict(test_loader, num_features=FEATURES, typ=t)

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

