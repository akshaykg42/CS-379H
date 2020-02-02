import argparse, pickle
import numpy as np
import os

import torch
from torch import nn, optim

# Use the util.load() function to load your dataset
from .utils import *
from .model import *

dirname = os.path.dirname(os.path.abspath(__file__))
sent_type = 0

def train(iterations, batch_size=16):
	'''
	This is the main training function.
	'''

	"""
	Load the training data
	"""

	train_inputs, train_labels = load('pcr_documents.pkl', 'pcr_summaries.pkl', 'pcr_oracles.pkl', sent_type)
	
	# TODO: Update loss
	loss = nn.CrossEntropyLoss()
	model = OracleSelectorModel().cuda()
	
	# TODO: Update optimizer
	# optimizer = optim.Adam(model.parameters(), lr = 1e-4)
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
	
	for iteration in range(iterations):
		model.train()
		# Construct a mini-batch
		batch = np.random.choice(train_inputs.shape[0], batch_size)
		'''
		This gives TypeError: can't convert np.ndarray of type numpy.object_. The only supported types are: float64, float32, float16, int64, int32, int16, int8, uint8, and bool.
		Since the input is irregular in shape (variable #sentences per document)
		'''
		batch_inputs = torch.as_tensor(train_inputs[batch], dtype=torch.float32).cuda()
		batch_labels = torch.as_tensor(train_labels[batch], dtype=torch.long).cuda()
		
		# zero the gradients (part of pytorch backprop)
		optimizer.zero_grad()
		
		# Compute the model output and loss (view flattens the input)
		# Need softmax output to calculate loss but model returns argmax(softmax) which is single value
		# If we return softmax instead then output_dim cannot be fixed
		loss_val = loss( model(batch_inputs), batch_labels)
		
		# Compute the gradient
		loss_val.backward()
		
		# Update the weights
		optimizer.step()
		
		if iteration % 10 == 0:
			print('[%5d]'%iteration, 'loss = %f'%loss_val)

	# Save the trained model
	torch.save(model.state_dict(), os.path.join(dirname, model_name + '.th')) # Do NOT modify this line


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--iterations', type=int, default=10000)
	args = parser.parse_args()

	print ('[I] Start training')
	train(args.iterations)
	print ('[I] Training finished')
