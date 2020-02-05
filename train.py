import argparse, pickle
import numpy as np
import os

import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split

# Use the util.load() function to load your dataset
from utils import *
from model import *

dirname = os.path.dirname(os.path.abspath(__file__))
model_name = 'OracleSelectorModel'
sent_type = 0

def pad_and_mask(batch_inputs):
	batch_size = len(batch_inputs)
	lengths = np.array([len(example) for example in batch_inputs])
	max_len = max(lengths)
	num_features = batch_inputs[0].shape[1]
	padded_inputs = np.zeros((batch_size, max_len, num_features))
	for i, example in enumerate(batch_inputs):
		for j, sentence in enumerate(example):
			padded_inputs[i][j] = sentence
	mask = np.arange(max_len) < lengths[:, None]
	padded_inputs = torch.from_numpy(padded_inputs).float().cuda()
	mask = (~(torch.from_numpy(mask).byte())).to(torch.bool).cuda()
	return mask, padded_inputs

def train(train_inputs, train_labels, iterations, batch_size=16):
	'''
	This is the main training function.
	'''

	"""
	Load the training data
	"""
	# with open('X_values.pkl', 'rb') as f:
	# 	train_inputs = np.load(f, allow_pickle=True)
	# with open('y_values.pkl', 'rb') as f:
	# 	train_labels = np.load(f, allow_pickle=True)
	num_features = train_inputs[0].shape[1]

	loss = nn.NLLLoss()
	model = OracleSelectorModel(num_features).cuda()

	# optimizer = optim.Adam(model.parameters(), lr = 1e-4)
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

	for iteration in range(iterations):
		model.train()
		# Construct a mini-batch
		batch = np.random.choice(train_inputs.shape[0], batch_size)
		batch_inputs = train_inputs[batch]
		batch_labels = torch.from_numpy(train_labels[batch]).unsqueeze(1).cuda()
		mask, padded_inputs = pad_and_mask(batch_inputs)
		
		# zero the gradients (part of pytorch backprop)
		optimizer.zero_grad()
		
		# Compute the model output and loss
		batch_scores, pred_labels = model(padded_inputs, mask)
		loss_val = loss(batch_scores, batch_labels)
		
		# Compute the gradient
		loss_val.backward()
		
		# Update the weights
		optimizer.step()
		
		if iteration % 10 == 0:
			print('[%5d]'%iteration, 'loss = %f'%loss_val)

	# Save the trained model
	torch.save(model.state_dict(), os.path.join(dirname, model_name + '.th')) # Do NOT modify this line

def test(test_inputs, test_labels):
	num_features = test_inputs[0].shape[1]
	
	model = OracleSelectorModel(num_features).cuda()
	model.load_state_dict(torch.load(os.path.join(dirname, model_name + '.th')))
	model.eval()

	mask, padded_inputs = pad_and_mask(test_inputs)
	test_scores, pred_labels = model(padded_inputs, mask)
	accuracy = len([i for i in range(len(test_labels)) if test_labels[i] == pred_labels[i].item()])/len(test_labels)
	print('Accuracy: {}'.format(accuracy))

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--iterations', type=int, default=1000)
	args = parser.parse_args()
	X, y = load('pcr_documents.pkl', 'pcr_summaries.pkl', 'pcr_oracles.pkl', sent_type)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	print ('[I] Start training')
	train(X_train, y_train, args.iterations)
	print ('[I] Training finished')
	test(X_test, y_test)

