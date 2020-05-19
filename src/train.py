#train -> specify dataset name, linear or bert, topic, batch size, # epochs, mini
#have functions for either in the same file trainer.py in models/
import os
import time
import datetime
import argparse
from torch import nn, optim
from models.linear import *
from utils.earlystopping import *
from models.data_loader import *
from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, AdamW, BertConfig

# Patience is the number of consecutive iterations with no improvement in validation loss before training is aborted
def train_linear(train_loader, valid_loader, n_epochs, batch_size, topic, patience=7):
	print ('[I] Start training')
	"""
	Load the training data
	"""

	# to track the training loss as the model trains
	train_losses = []
	# to track the validation loss as the model trains
	valid_losses = []
	# to track the average training loss per epoch as the model trains
	avg_train_losses = []
	# to track the average validation loss per epoch as the model trains
	avg_valid_losses = []

	# Get number of features
	for inputs, mask, targets in train_loader:
		num_features = inputs[0].shape[1]
		break

	loss = nn.NLLLoss()
	model = LinearModel(num_features).cuda()
	early_stopping = EarlyStopping(train_loader.dataset.dataset_name, patience=patience, verbose=True)

	optimizer = optim.Adam(model.parameters(), lr = 1e-2)
	# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

	# Train for some epochs
	for epoch in range(1, n_epochs + 1):
		model.train()

		# Get batches from training dataloader
		for batch, (inputs, mask, targets) in enumerate(train_loader):
			optimizer.zero_grad()
			scores, preds = model(inputs, mask)
			train_loss = loss(scores, targets)
			train_loss.backward()
			optimizer.step()
			train_losses.append(train_loss.item())

		model.eval()

		# Get batches from validation dataloader
		for inputs, mask, targets in valid_loader:
			scores, preds = model(inputs, mask)
			valid_loss = loss(scores, targets)
			valid_losses.append(valid_loss.item())

		# Calculate training/validation loss at each epoch
		train_loss = np.average(train_losses)
		valid_loss = np.average(valid_losses)
		avg_train_losses.append(train_loss)
		avg_valid_losses.append(valid_loss)

		epoch_len = len(str(n_epochs))
		
		# Logging
		print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
					 f'train_loss: {train_loss:.5f} ' +
					 f'valid_loss: {valid_loss:.5f}')

		print(print_msg)
		
		# Clear lists to track next epoch
		train_losses = []
		valid_losses = []
		
		# early_stopping needs the validation loss to check if it has decresed, 
		# and if it has, it will make a checkpoint of the current model
		early_stopping(valid_loss, model)
		
		if early_stopping.early_stop:
			print("Early stopping")
			break

	print ('[I] Training finished')

	save_path = '../models/{}/linear'.format(train_loader.dataset.dataset_name)

	# Load last checkpoint and save it
	model.load_state_dict(torch.load('{}/checkpoint.pt'.format(save_path)))

	print("Saving model to %s" % '{}/{}.th'.format(save_path, topic))
	torch.save(model.state_dict(), '{}/{}.th'.format(save_path, topic))

def flat_accuracy(preds, labels):
	pred_flat = np.argmax(preds, axis=0).flatten()
	labels_flat = labels.flatten()
	return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
	'''
	Takes a time in seconds and returns a string hh:mm:ss
	'''
	# Round to the nearest second.
	elapsed_rounded = int(round((elapsed)))
	# Format as hh:mm:ss
	return str(datetime.timedelta(seconds=elapsed_rounded))

def train_bert(train_loader, valid_loader, n_epochs, batch_size, topic):
	device = torch.device("cuda")

	criterion = nn.CrossEntropyLoss()

	model = BertForSequenceClassification.from_pretrained(
		"bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
		num_labels = 1, # The number of output labels--2 for binary classification.
						# You can increase this for multi-class tasks.   
		output_attentions = False, # Whether the model returns attentions weights.
		output_hidden_states = False, # Whether the model returns all hidden-states.
	).cuda()

	optimizer = AdamW(model.parameters(),
		lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
		eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
	)

	total_steps = len(train_loader) * n_epochs
	scheduler = get_linear_schedule_with_warmup(
		optimizer, 
		num_warmup_steps = 0, # Default value in run_glue.py
		num_training_steps = total_steps
	)

	# Store the average loss after each epoch so we can plot them.
	loss_values = []

	# For each epoch...
	for epoch_i in range(0, n_epochs):
		
		# ========================================
		#			   Training
		# ========================================
		
		# Perform one full pass over the training set.

		print("")
		print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, n_epochs))
		print('Training...')

		# Measure how long the training epoch takes.
		t0 = time.time()

		# Reset the total loss for this epoch.
		total_loss = 0

		# Put the model into training mode. Don't be mislead--the call to 
		# `train` just changes the *mode*, it doesn't *perform* the training.
		# `dropout` and `batchnorm` layers behave differently during training
		# vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
		model.train()

		# For each batch of training data...
		for step, batch in enumerate(train_loader):

			# Progress update every 10 batches.
			if step % 10 == 0 and not step == 0:
				# Calculate elapsed time in minutes.
				elapsed = format_time(time.time() - t0)
				
				# Report progress.
				print('  Batch {:>5,}  of  {:>5,}.	Elapsed: {:}.'.format(step, len(train_loader), elapsed))

			# Unpack this training batch from our dataloader. 
			#
			# As we unpack the batch, we'll also copy each tensor to the GPU using the 
			# `to` method.
			#
			# `batch` contains three pytorch tensors:
			#   [0]: input ids 
			#   [1]: attention masks
			#   [2]: labels 
			b_input_ids = batch[0].to(device)
			b_input_mask = batch[1].to(device)
			b_labels = batch[2].to(device)
			b_lens = batch[3]

			splits = [0]
			splits.extend(list(accumulate(b_lens)))

			# Always clear any previously calculated gradients before performing a
			# backward pass. PyTorch doesn't do this automatically because 
			# accumulating the gradients is "convenient while training RNNs". 
			# (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
			model.zero_grad()		

			# Perform a forward pass (evaluate the model on this training batch).
			# This will return the loss (rather than the model output) because we
			# have provided the `labels`.
			# The documentation for this `model` function is here: 
			# https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
			outputs = model(b_input_ids, 
						token_type_ids=None, 
						attention_mask=b_input_mask)
			
			logits = outputs[0].unsqueeze(dim=0)
			loss = criterion(logits, b_labels)

			# Accumulate the training loss over all of the batches so that we can
			# calculate the average loss at the end. `loss` is a Tensor containing a
			# single value; the `.item()` function just returns the Python value 
			# from the tensor.
			total_loss += loss.item()

			# Perform a backward pass to calculate the gradients.
			loss.backward()

			# Clip the norm of the gradients to 1.0.
			# This is to help prevent the "exploding gradients" problem.
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

			# Update parameters and take a step using the computed gradient.
			# The optimizer dictates the "update rule"--how the parameters are
			# modified based on their gradients, the learning rate, etc.
			optimizer.step()

			# Update the learning rate.
			scheduler.step()

		# Calculate the average loss over the training data.
		avg_train_loss = total_loss / len(train_loader)			
		
		# Store the loss value for plotting the learning curve.
		loss_values.append(avg_train_loss)

		print("")
		print("  Average training loss: {0:.2f}".format(avg_train_loss))
		print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
			
		# ========================================
		#			   Validation
		# ========================================
		# After the completion of each training epoch, measure our performance on
		# our validation set.

		print("")
		print("Running Validation...")

		t0 = time.time()

		# Put the model in evaluation mode--the dropout layers behave differently
		# during evaluation.
		model.eval()

		# Tracking variables 
		eval_loss, eval_accuracy = 0, 0
		nb_eval_steps, nb_eval_examples = 0, 0

		# Evaluate data for one epoch
		for batch in valid_loader:
			
			# Add batch to GPU
			b_input_ids = batch[0].to(device)
			b_input_mask = batch[1].to(device)
			b_labels = batch[2].to(device)
			b_lens = batch[3]
			
			# Telling the model not to compute or store gradients, saving memory and
			# speeding up validation
			with torch.no_grad():		

				# Forward pass, calculate logit predictions.
				# This will return the logits rather than the loss because we have
				# not provided labels.
				# token_type_ids is the same as the "segment ids", which 
				# differentiates sentence 1 and 2 in 2-sentence tasks.
				# The documentation for this `model` function is here: 
				# https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
				outputs = model(b_input_ids, 
								token_type_ids=None, 
								attention_mask=b_input_mask)
			
			# Get the "logits" output by the model. The "logits" are the output
			# values prior to applying an activation function like the softmax.
			logits = outputs[0]

			# Move logits and labels to CPU
			logits = logits.detach().cpu().numpy()
			label_ids = b_labels.to('cpu').numpy()
			
			# Calculate the accuracy for this batch of test sentences.
			tmp_eval_accuracy = flat_accuracy(logits, label_ids)
			
			# Accumulate the total accuracy.
			eval_accuracy += tmp_eval_accuracy

			# Track the number of batches
			nb_eval_steps += 1

		# Report the final accuracy for this validation run.
		print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
		print("  Validation took: {:}".format(format_time(time.time() - t0)))

	print("")
	print("Training complete!")

	save_path = '../models/{}/bert/{}/'.format(train_loader.dataset.dataset_name, topic)

	if not os.path.exists(save_path):
		os.makedirs(save_path)

	print("Saving model to %s" % save_path)

	model.save_pretrained(save_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-dataset_name')
	parser.add_argument('-model_type')
	# Models are trained per topic
	parser.add_argument('-topic', type=int)
	'''
	Recommended batch sizes:
	BERT: 1
	Linear: 16
	'''
	parser.add_argument('-batch_size', type=int)
	'''
	Recommended number of epochs:
	BERT: 4
	Linear: 100
	'''
	parser.add_argument('-epochs', type=int)
	'''
	If BERT runs out of memory even with a batch size of 1, try using mini
	This will "reduce" each document to K(=10) sentences including the oracle
	For more details, seee src/models/data_loader.py
	'''
	parser.add_argument('-m', '--mini', action='store_true', default=False)

	args = parser.parse_args()

	dataset_name, model_type, topic, batch_size, epochs, mini = \
		args.dataset_name, args.model_type, args.topic, args.batch_size, args.epochs, args.mini

	train_loader = create_loader(dataset_name, model_type, 'train', topic, batch_size, mini)
	valid_loader = create_loader(dataset_name, model_type, 'val', topic, batch_size, mini)

	save_dir = '../models/{}/{}/'.format(dataset_name, model_type)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	# Set training function based on model type
	if(model_type == 'linear'):
		train_fn = train_linear
	elif(model_type == 'bert'):
		train_fn = train_bert

	train_fn(train_loader, valid_loader, epochs, batch_size, topic)








