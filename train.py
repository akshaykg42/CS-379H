from utils import *
from model import *
from earlystopping import *
import os
import argparse, pickle
from torch import nn, optim

dirname = os.path.dirname(os.path.abspath(__file__))
model_name = 'OracleSelectorModel'

def train(train_loader, valid_loader, n_epochs, batch_size, patience=7):
	'''
	This is the main training function.
	'''
	print ('[I] Start training')
	"""
	Load the training data
	"""
	num_features = 4127

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

	optimizer = optim.Adam(model.parameters(), lr = 1e-2)
	# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

	for epoch in range(1, n_epochs + 1):
		model.train()

		for batch, (inputs, mask, targets) in enumerate(train_loader):
			optimizer.zero_grad()
			scores, preds = model(inputs, mask)
			train_loss = loss(scores, targets)
			train_loss.backward()
			optimizer.step()
			train_losses.append(train_loss.item())

		model.eval()

		for inputs, mask, targets in valid_loader:
			scores, preds = model(inputs, mask)
			valid_loss = loss(scores, targets)
			valid_losses.append(valid_loss.item())

		train_loss = np.average(train_losses)
		valid_loss = np.average(valid_losses)
		avg_train_losses.append(train_loss)
		avg_valid_losses.append(valid_loss)

		epoch_len = len(str(n_epochs))
		
		print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
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

