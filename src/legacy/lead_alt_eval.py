from utils import *
import math
import argparse
import os
from collections import Counter

dirname = os.path.dirname(os.path.abspath(__file__))
model_name = 'linear_model_quakes'

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--datadir', default='pcr_data')

	args = parser.parse_args()

	DATA_DIR = args.datadir

	print('Loading data...')
	documents, summaries, oracles, types = load(DATA_DIR)
	test_files = os.listdir(DATA_DIR + '/processed/test/')
	test_indices = [int(file[:-4]) for file in test_files if file.endswith('.npy')]
	summaries = [summaries[i] for i in test_indices]
	documents = [documents[i] for i in test_indices]
	types = [types[i] for i in test_indices]
	oracles = [oracles[i] for i in test_indices]

	scores = []
	
	for i, t in enumerate(types):
		d = dict(Counter([x[0] for x in t]))
		nongarbage = [i for i in range(len(t)) if t[i][0] != 6]
		oracle = [oracles[i][j] for j in nongarbage]
		t_ = [t[j][0] for j in nongarbage]
		score = 0.0
		for j in range(len(nongarbage)):
			pos = oracle[j] + 1
			score += d[t_[j]]/len(documents[i])
			d[t_[j]] -= 1
			score -= pos/len(documents[i])
		scores.append(score/len(documents[i]))
	print(np.mean(scores))

