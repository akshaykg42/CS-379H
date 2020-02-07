from utils import *
from model import *
from earlystopping import *
import os
import argparse, pickle
from torch import nn, optim

dirname = os.path.dirname(os.path.abspath(__file__))
model_name = 'OracleSelectorModel'

def train(train_inputs, train_labels, val_inputs, val_labels, patience=20, iterations=1000, batch_size=16):
	'''
	This is the main training function.
	'''
	print ('[I] Start training')
	"""
	Load the training data
	"""
	num_features = train_inputs[0].shape[1]

	# to track the training loss as the model trains
	train_losses = []
	# to track the validation loss as the model trains
	valid_losses = []
	# to track the average training loss per epoch as the model trains
	avg_train_losses = []
	# to track the average validation loss per epoch as the model trains
	avg_valid_losses = [] 

	loss = nn.NLLLoss()
	model = OracleSelectorModel(num_features).cuda()
	early_stopping = EarlyStopping(patience=patience, verbose=True)
	val_labels = torch.from_numpy(val_labels).unsqueeze(1).cuda()

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
		train_loss = loss(batch_scores, batch_labels)
		train_losses.append(train_loss.item())
		
		# Compute the gradient
		train_loss.backward()
		
		# Update the weights
		optimizer.step()

		# Prep model for evaluation
		model.eval()

		# Compute the validation output and loss
		val_mask, padded_val_inputs = pad_and_mask(val_inputs)
		val_scores, val_pred_labels = model(padded_val_inputs, val_mask)
		val_loss = loss(val_scores, val_labels)
		valid_losses.append(val_loss.item())

		# Print training/validation statistics 
		# Calculate average loss over an epoch
		train_loss = np.average(train_losses)
		valid_loss = np.average(valid_losses)
		avg_train_losses.append(train_loss)
		avg_valid_losses.append(valid_loss)

		iter_len = len(str(iterations))
		
		print_msg = (f'[{iteration:>{iter_len}}/{iterations:>{iter_len}}] ' +
					 f'train_loss: {train_loss:.5f} ' +
					 f'valid_loss: {valid_loss:.5f}')

		print(print_msg)

		# clear lists to track next epoch
		train_losses = []
		valid_losses = []

		# early_stopping needs the validation loss to check if it has decresed, 
		# and if it has, it will make a checkpoint of the current model
		early_stopping(valid_loss, model)

		if early_stopping.early_stop:
			print("Early stopping")
			break

	print ('[I] Training finished')

	model.load_state_dict(torch.load('checkpoint.pt'))
	# Save the trained model
	torch.save(model.state_dict(), os.path.join(dirname, model_name + '.th')) # Do NOT modify this line

