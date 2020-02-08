from model import *
from utils import *
import os

dirname = os.path.dirname(os.path.abspath(__file__))
model_name = 'OracleSelectorModel'

def test(test_loader):
	print('[II] Start testing')
	num_features = test_inputs[0].shape[1]
	
	model = OracleSelectorModel(num_features).cuda()
	model.load_state_dict(torch.load(os.path.join(dirname, model_name + '.th')))
	model.eval()

	accuracy = []
	for inputs, mask, targets in test_loader:
		scores, preds = model(inputs, mask)
		for i, pred in enumerate(preds):
			accuracy.append(pred == targets[i])
	accuracy = sum(accuracy)/len(accuracy)
	print('[II] Accuracy: {}'.format(accuracy))
	return scores