from model import *
from utils import *
import os

dirname = os.path.dirname(os.path.abspath(__file__))
model_name = 'OracleSelectorModel'

def test(test_loader):
	print('[II] Start testing')
	num_features = 4127
	
	model = OracleSelectorModel(num_features).cuda()
	model.load_state_dict(torch.load(os.path.join(dirname, model_name + '.th')))
	model.eval()

	accuracy = []
	for inputs, mask, targets in test_loader:
		scores, preds = model(inputs, mask)
		for i, pred in enumerate(preds):
			if(pred == targets[i]):
				accuracy.append(1)
			else:
				accuracy.append(0)
	print(len(accuracy))
	accuracy = sum(accuracy)/len(accuracy)
	print('[II] Accuracy: {}'.format(accuracy))
	return scores
