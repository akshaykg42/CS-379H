from model import *
from utils import *
import os

dirname = os.path.dirname(os.path.abspath(__file__))
model_name = 'linear_model_pcr'

def test(test_loader, num_features, typ):
	print('[II] Start testing')
	
	model = OracleSelectorModel(num_features).cuda()
	model.load_state_dict(torch.load(os.path.join(dirname, model_name + str(typ) + '.th')))
	model.eval()

	all_scores = []
	accuracy = []
	for inputs, mask, targets in test_loader:
		scores, preds = model(inputs, mask)
		for i, pred in enumerate(preds):
			if(pred == targets[i]):
				accuracy.append(1)
			else:
				accuracy.append(0)
			all_scores.append(scores)
	accuracy = sum(accuracy)/len(accuracy)
	print('[II] Accuracy: {}'.format(accuracy))
	return all_scores

