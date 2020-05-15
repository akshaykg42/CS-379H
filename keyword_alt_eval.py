from utils import *
import math
import argparse
import os
from collections import Counter

keywords_courtlistener = [["petition", "appeals", "relief", "denial", "pleaded", "guilty"], ["argues", "claims", "contends", "filed"], ["conclude", "failed", "review", "abuse", "determined"], ["review", "affirmed", "affirm", "reverse", "reversed"]]
keywords = [["magnitude", "earthquake", "quake", "earthquakes", "quakes", "strike", "struck", "aftershock", "aftershocks", "depth", "tsunami", "tremor", "tremors", "hit"],["epicenter", "north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest", "located", "miles", "km", "kilometers", "felt", "centered"],["destroyed", "collapse", "collapses", "collapsed", "building", "buildings", "road", "roads", "power", "devastated", "devastation", "flattened", "disrupted", "rubble", "fire"],["dead", "injured", "injuries", "death", "killed", "fatalities", "loss", "toll", "casualties", "buried", "trapped", "outbreak", "missing"],["a.m.", "p.m.", "gmt", "et", "pt", "hours", "time", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"],["planes", "aid", "help", "helping", "warning", "warnings", "response", "rescue", "evacuate", "evacuation", "team", "teams", "supplies", "emergency", "reach", "reached", "reaching", "donation", "donate", "search", "searching", "affected", "disaster", "recover", "recovery", "survivors", "charity", "$", "hospital", "responders", "relief"]]
keywords = [set(x) for x in keywords]

dirname = os.path.dirname(os.path.abspath(__file__))
model_name = 'linear_model_quakes'
TYPES = [0, 1, 2, 3, 4, 5]

def get_type_score(sentence, t):
	sentence = set([word.lower() for word in nltk.word_tokenize(sentence)])
	score = len(sentence.intersection(keywords[t]))
	return score

def predict(document, TYPE):
	scores = [get_type_score(sentence, TYPE) for sentence in document]
	highscore = max(scores)
	if(highscore == 0):
		winner = list(range(len(document)))
		random.shuffle(winner)
	else:
		winner = list(np.argsort(scores)[::-1])
	return sorted(scores)[::-1], winner

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
			modelscores, pred = predict(documents[i], t_[j])
			if(max(modelscores) != 0):
				pos = pred.index(oracle[j]) + 1
			else:
				pos = pred.index(oracle[j]) + 1
			score += d[t_[j]]/len(documents[i])
			d[t_[j]] -= 1
			score -= pos/len(documents[i])
		scores.append(score/len(documents[i]))
	print(np.mean(scores))

