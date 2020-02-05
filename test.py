from model import *
from utils import *
import os

dirname = os.path.dirname(os.path.abspath(__file__))
model_name = 'OracleSelectorModel'

def test(test_inputs, test_labels):
	print('[II] Start testing')
	num_features = test_inputs[0].shape[1]
	
	model = OracleSelectorModel(num_features).cuda()
	model.load_state_dict(torch.load(os.path.join(dirname, model_name + '.th')))
	model.eval()

	mask, padded_inputs = pad_and_mask(test_inputs)
	test_scores, pred_labels = model(padded_inputs, mask)
	accuracy = len([i for i in range(len(test_labels)) if test_labels[i] == pred_labels[i].item()])/len(test_labels)
	print('[II] Accuracy: {}'.format(accuracy))