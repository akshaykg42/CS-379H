from utils import *
import argparse

keywords = [["magnitude", "earthquake", "quake", "earthquakes", "quakes", "strike", "struck", "aftershock", "aftershocks", "depth", "tsunami", "tremor", "tremors", "hit"],["epicenter", "north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest", "located", "miles", "km", "kilometers", "felt", "centered"],["destroyed", "collapse", "collapses", "collapsed", "building", "buildings", "road", "roads", "power", "devastated", "devastation", "flattened", "disrupted", "rubble", "fire"],["dead", "injured", "injuries", "death", "killed", "fatalities", "loss", "toll", "casualties", "buried", "trapped", "outbreak", "missing"],["a.m.", "p.m.", "gmt", "et", "pt", "hours", "time", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"],["planes", "aid", "help", "helping", "warning", "warnings", "response", "rescue", "evacuate", "evacuation", "team", "teams", "supplies", "emergency", "reach", "reached", "reaching", "donation", "donate", "search", "searching", "affected", "disaster", "recover", "recovery", "survivors", "charity", "$", "hospital", "responders", "relief"]]
keywords = [set(x) for x in keywords]

def get_type_score(sentence, t):
	sentence = set([word.lower() for word in nltk.word_tokenize(sentence)])
	score = len(sentence.intersection(keywords[t]))
	return score


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', '--type', type=int, default=0)
	parser.add_argument('-d', '--datadir', default='pcr_data')

	args = parser.parse_args()

	DATA_DIR, TYPE = args.datadir, args.type

	print('Loading data...')
	documents, summaries, oracles, types = load(DATA_DIR)
	test_indices = [6, 7, 9, 13, 14, 17, 18, 37, 47, 54, 59, 62, 63, 68, 73, 89, 91, 108, 114, 127, 128, 132, 138, 145, 152, 157, 158, 160, 167, 170, 172, 173, 175, 185, 188, 191, 193, 199, 202, 207, 208, 212, 214, 217, 238, 239, 249, 259, 260, 262, 264, 268, 272, 278, 279, 280, 287, 289, 297, 306, 307, 311, 313, 321, 323, 325, 326, 340, 342, 345, 347, 350, 358, 359, 361, 374, 378, 391, 394, 395, 398, 409, 410, 415, 423, 424, 428, 429, 439, 442, 443, 444, 447, 449, 451, 459, 465, 467, 469, 471, 474, 477, 478, 481, 488, 497, 507, 509, 514, 523, 535, 558, 560, 572, 576, 577]
	documents = [documents[i] for i in test_indices]
	summaries = [summaries[i] for i in test_indices]
	oracles = [oracles[i] for i in test_indices]
	types = [types[i] for i in test_indices]

	correct = 0.0
	total = 0.0

	for i in range(len(documents)):
		document = documents[i]
		summary_indices_type = [j for j in range(len(types[i])) if TYPE in types[i][j]]
		oracle_indices_type = [oracles[i][j] for j in summary_indices_type]
		if(len(summary_indices_type) > 0):
			total += 1
			scores = [get_type_score(sentence, TYPE) for sentence in document]
			highscore = max(scores)
			if(highscore == 0):
				continue
			winners = [np.argmax(scores)]
			#winners = [i for i in range(len(scores)) if scores[i] == highscore]
			if(set(winners).intersection(set(oracle_indices_type))):
				correct += 1
			print('Model selected sentences:\n')
			for winner in winners:
				print(document[winner])
				print()
			print()
			print('Ground Truth sentences:\n')
			for j in oracle_indices_type:
				print(document[j])
				print()
			print()
			print('---------------------------------------------------------')
	print(correct)
	print(total)
	print(correct/total)
