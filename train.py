from utils import *
from model import *
import os
import argparse, pickle
from torch import nn, optim

dirname = os.path.dirname(os.path.abspath(__file__))
model_name = 'OracleSelectorModel'

def train(train_inputs, train_labels, iterations=1000, batch_size=16):
	'''
	This is the main training function.
	'''
	print ('[I] Start training')
	"""
	Load the training data
	"""
	num_features = train_inputs[0].shape[1]

	loss = nn.NLLLoss()
	model = OracleSelectorModel(num_features).cuda()

	optimizer = optim.Adam(model.parameters(), lr = 1e-2)
	# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

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
	print ('[I] Training finished')
	# Save the trained model
	torch.save(model.state_dict(), os.path.join(dirname, model_name + '.th')) # Do NOT modify this line

